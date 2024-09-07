# -*- coding: utf-8 -*-
# @Time    : 6/30/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : generate.py

"""
Batch Generator for LLMs.
Our optimization objective is primarily throughput, which is used to introduce generation in the training process.
"""

__all__ = [
    "Generator",
    "HFGenerator",
    "HFPipelineGenerator",
    "VLLMGenerator",
    "VLLMGeneratorFromLora"
]

import copy
import itertools
import logging
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Type

import math
import ray
import torch
import tqdm
from peft import PeftModel
from pydantic import BaseModel
from transformers import PreTrainedTokenizer, pipeline, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, RequestOutput

from mm_video.utils.common.data import chunk

logger = logging.getLogger(__name__)


class Generator(ABC):
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> Union[List[str], List[List[str]]]:
        pass


class HFGenerator(Generator):
    def __init__(
            self,
            model_name_or_path: str,
            model_init_kwargs: Optional[dict] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            batch_size: int = 1,
            torch_compile: bool = False,
            generate_kwargs: Optional[dict] = None,
    ):
        """
        Basic generator based on HuggingFace GenerationMixin.
        Only support running on single GPU, do not support tensor parallel or data parallel.
        Do not support beam search or multinomial sampling (TBD).

        :param model_name_or_path:
        :param model_init_kwargs:
        :param tokenizer:
        :param batch_size: Control batching when generating.
        :param torch_compile:
        :param generate_kwargs: Kwargs for HuggingFace generation.
        """
        self.batch_size = batch_size

        if model_init_kwargs is None:
            model_init_kwargs = {}
        if generate_kwargs is None:
            generate_kwargs = {}
        self.default_generate_kwargs = generate_kwargs

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_init_kwargs)
        self.model.eval()
        self.model.cuda()
        if torch_compile:
            self.model = torch.compile(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer

        if self.tokenizer.pad_token_id is None:
            logger.warning("Setting pad_token_id to eos_token_id for generation.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate_batch(self, prompts: List[str], **generate_kwargs) -> List[str]:
        # Temporally change padding side to 'left' and tokenize
        padding_side_before = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        self.tokenizer.padding_side = padding_side_before

        kwargs = copy.deepcopy(self.default_generate_kwargs)
        kwargs.update(generate_kwargs)

        seq_len = input_ids.size(1)
        # noinspection PyUnresolvedReferences
        output_ids = self.model.generate(
            input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            use_cache=True,
            **kwargs
        )[:, seq_len:]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def generate(self, prompts: List[str], **generate_kwargs) -> List[str]:
        # Batching
        prompt_batches = []
        for i in range(0, len(prompts), self.batch_size):
            prompt_batches.append(prompts[i:i + self.batch_size])

        responses = []
        for prompt_batch in tqdm.tqdm(prompt_batches, desc="Generating"):
            responses.extend(self._generate_batch(prompt_batch, **generate_kwargs))
        # TODO: Process beam search output
        return responses


class HFPipelineGenerator(Generator):
    def __init__(
            self,
            model_name_or_path: str,
            model_init_kwargs: Optional[dict] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            batch_size: int = 1,
            pipeline_kwargs: Optional[dict] = None,
            generate_kwargs: Optional[dict] = None,
            json_schema: Optional[Type[BaseModel]] = None
    ):
        """
        Generator based on HuggingFace pipeline.
        Support pipeline parallel, do not support tensor parallel.

        :param model_name_or_path:
        :param model_init_kwargs:
        :param tokenizer:
        :param batch_size:
        :param pipeline_kwargs:
        :param generate_kwargs:
        :param json_schema: Json schema for lm-format-enforcer.
        """
        if pipeline_kwargs is None:
            pipeline_kwargs = {}
        if generate_kwargs is None:
            generate_kwargs = {}

        self.default_generate_kwargs = generate_kwargs
        self.batch_size = batch_size

        @ray.remote(num_gpus=1)
        class GenerateWorker:
            def __init__(self):
                self.pipeline = pipeline(
                    model=model_name_or_path,
                    tokenizer=model_name_or_path if tokenizer is None else tokenizer,
                    task="text-generation",
                    model_kwargs=model_init_kwargs,
                    batch_size=batch_size,
                    return_full_text=False,
                    **pipeline_kwargs
                )

                if self.pipeline.tokenizer.pad_token_id is None:
                    logger.warning("Setting pad_token_id to eos_token_id for generation.")
                    self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id

                if json_schema is not None:
                    from lmformatenforcer import JsonSchemaParser
                    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

                    parser = JsonSchemaParser(json_schema.schema())
                    prefix_function = build_transformers_prefix_allowed_tokens_fn(self.pipeline.tokenizer, parser)
                    generate_kwargs.update({"prefix_allowed_tokens_fn": prefix_function})

            def generate(self, prompts, kwargs):
                return self.pipeline(prompts, **kwargs)

        self.actors = [GenerateWorker.remote() for _ in range(torch.cuda.device_count())]
        self.actor_pool = ray.util.ActorPool(self.actors)

    def get_text(self, data: Union[list, dict]) -> Union[list, str]:
        if type(data) is list:
            return [self.get_text(x) for x in data]
        elif type(data) is dict:
            return data["generated_text"]
        else:
            raise ValueError(f"Unexpected type {type(data)}")

    def generate(self, prompts: List[str], use_tqdm: bool = True, **kwargs) -> Union[List[str], List[List[str]]]:
        generate_kwargs = copy.deepcopy(self.default_generate_kwargs)
        generate_kwargs.update(kwargs)

        # Chunk by batch size for the maximum throughput
        chunked_prompts = chunk(prompts, chunk_size=self.batch_size)

        ret = list(tqdm.tqdm(
            itertools.chain.from_iterable(
                self.actor_pool.map(
                    lambda a, v: a.generate.remote(*v),
                    [(p, generate_kwargs) for p in chunked_prompts]
                )
            ),
            desc="Generating", total=len(prompts), disable=not use_tqdm
        ))

        ret = self.get_text(ret)
        if all(len(x) == 1 for x in ret):
            ret = [x[0] for x in ret]
        return ret


class VLLMGenerator(Generator):
    def __init__(
            self,
            model_name_or_path: str,
            model_init_kwargs: Optional[dict] = None,
            tokenizer: Optional[str] = None,
            sampling_params_kwargs: Optional[dict] = None,
            generate_kwargs: Optional[dict] = None,
            json_schema: Optional[Type[BaseModel]] = None,
    ):
        """
        Running generate with vLLM. Will use all GPU with tensor parallel and data parallel enabled.

        :param model_name_or_path:
        :param model_init_kwargs: Additional kwargs for `vllm.LLM.__init__`.
        :param tokenizer: Tokenizer, will load a new tokenizer from the model_name_or_path if not provided.
        :param sampling_params_kwargs: Kwargs for `vllm.SamplingParams`.
        :param generate_kwargs: Kwargs for `vllm.LLM.generate`.
        :param json_schema: Json schema for lm-format-enforcer.
        """
        if model_init_kwargs is None:
            model_init_kwargs = {}
        if sampling_params_kwargs is None:
            sampling_params_kwargs = {}
        if generate_kwargs is None:
            generate_kwargs = {}

        self.default_sampling_params_kwargs = sampling_params_kwargs
        self.default_generate_kwargs = generate_kwargs

        # configure lm-format-enforcer
        if json_schema is not None:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, \
                build_vllm_token_enforcer_tokenizer_data

            tokenizer_data = build_vllm_token_enforcer_tokenizer_data(
                AutoTokenizer.from_pretrained(tokenizer) if tokenizer is not None else
                AutoTokenizer.from_pretrained(model_name_or_path)
            )
            parser = JsonSchemaParser(json_schema.schema())
            logits_processor = build_vllm_logits_processor(tokenizer_data, parser)
            sampling_params_kwargs["logits_processors"] = [logits_processor]

        class VLLMGenerateActor:
            def __init__(self, model_name_or_path, tokenizer, model_init_kwargs):
                self.model = LLM(
                    model=model_name_or_path,
                    tokenizer=tokenizer,
                    **model_init_kwargs
                )

            def generate(
                    self,
                    prompts: List[str],
                    generate_kwargs: dict,
                    sampling_params_kwargs: dict
            ) -> List[RequestOutput]:
                if "sampling_params" in generate_kwargs:
                    if sampling_params_kwargs:
                        logger.warning(
                            "Values in sampling_params_kwargs will be ignored because sampling_params is specified in "
                            "generate_kwargs."
                        )
                    return self.model.generate(prompts, **generate_kwargs)
                else:
                    sampling_params = SamplingParams(**sampling_params_kwargs)
                    return self.model.generate(prompts, sampling_params, **generate_kwargs)

        # Set `tensor_parallel_size` to 1 if not specified in `model_init_kwargs`.
        if "tensor_parallel_size" in model_init_kwargs:
            self.tensor_parallel_size = model_init_kwargs["tensor_parallel_size"]
        else:
            self.tensor_parallel_size = 1
        # Set `pipeline_parallel_size` if not specified in `model_init_kwargs`.
        if "pipeline_parallel_size" in model_init_kwargs:
            self.pipeline_parallel_size = model_init_kwargs["pipeline_parallel_size"]
        else:
            # If not specified, we use all GPUs available.
            self.pipeline_parallel_size = torch.cuda.device_count() // self.tensor_parallel_size
            model_init_kwargs["pipeline_parallel_size"] = self.pipeline_parallel_size

        if self.pipeline_parallel_size == 1:
            logger.debug("Pipeline parallel is disabled since we have `pipeline_parallel_size=1`. Only start a single"
                         "`LLM` instance to avoid any error from using `ray`.")

            # Pop out pipeline parallel size, since this argument is not supported in previous version of vLLM
            if "pipeline_parallel_size" in model_init_kwargs:
                model_init_kwargs.pop("pipeline_parallel_size")

            self.actors = VLLMGenerateActor(model_name_or_path, tokenizer, model_init_kwargs)
        else:
            logger.debug("Using custom pipeline parallel implementation for vLLM.")

            # Pop out pipeline parallel size, since this argument is not supported in previous version of vLLM
            if "pipeline_parallel_size" in model_init_kwargs:
                model_init_kwargs.pop("pipeline_parallel_size")

            # This is a workaround for running pipeline parallel with early version of vLLM.
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # Got "RuntimeError: No CUDA GPUs are available" if tp == 1 and resources are not set in ray.remote
            # See https://github.com/vllm-project/vllm/issues/973
            @ray.remote(num_gpus=1)
            class _VLLMGenerateActorOnOneGPU(VLLMGenerateActor):
                pass

            @ray.remote
            class _VLLMGenerateActorOnMultipleGPU(VLLMGenerateActor):
                pass

            if self.tensor_parallel_size == 1:
                # noinspection PyUnresolvedReferences
                self.actors = [
                    _VLLMGenerateActorOnOneGPU.remote(model_name_or_path, tokenizer, model_init_kwargs)
                    for _ in range(self.pipeline_parallel_size)
                ]
            else:
                self.actors = [
                    _VLLMGenerateActorOnMultipleGPU.remote(model_name_or_path, tokenizer, model_init_kwargs)
                    for _ in range(self.pipeline_parallel_size)
                ]

        logger.debug("vLLM generator tensor_parallel_size: %s", self.tensor_parallel_size)
        logger.debug("vLLM generator pipeline_parallel_size: %s", self.pipeline_parallel_size)

    def ray_batch(self, args_list, ordered: bool = True):
        if isinstance(self.actors, list):
            actor_pool = ray.util.ActorPool(self.actors)
            if ordered:
                return actor_pool.map(lambda a, v: a.generate.remote(*v), args_list)
            else:
                return actor_pool.map_unordered(lambda a, v: a.generate.remote(*v), args_list)
        else:
            return [self.actors.generate(*args) for args in args_list]

    @staticmethod
    def split_list(full_list, n_chunks):
        chunk_size = math.ceil(len(full_list) / n_chunks)
        return [full_list[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]

    def generate(
            self,
            prompts: List[str],
            generate_kwargs: Optional[dict] = None,
            ordered: bool = True,
            **sampling_params_kwargs
    ) -> Union[List[str], List[List[str]]]:
        if generate_kwargs is None:
            generate_kwargs = {}

        # Make it compatible with kwargs for HuggingFace generator
        if "max_new_tokens" in sampling_params_kwargs:
            sampling_params_kwargs["max_tokens"] = sampling_params_kwargs["max_new_tokens"]
            sampling_params_kwargs.pop("max_new_tokens")

        new_sampling_params_kwargs = copy.deepcopy(self.default_sampling_params_kwargs)
        new_sampling_params_kwargs.update(sampling_params_kwargs)

        new_generate_kwargs = copy.deepcopy(self.default_generate_kwargs)
        new_generate_kwargs.update(generate_kwargs)

        tp_size = self.tensor_parallel_size
        dp_size = torch.cuda.device_count() // tp_size
        split_prompts = self.split_list(prompts, dp_size)
        assert len(split_prompts) == dp_size

        results: List[RequestOutput] = list(itertools.chain.from_iterable(
            self.ray_batch(
                [(p, new_generate_kwargs, new_sampling_params_kwargs) for p in split_prompts],
                ordered=ordered
            )
        ))
        assert len(results) == len(prompts), "Length mismatch: {} vs {}".format(len(results), len(prompts))

        # Get text
        ret = [[c.text for c in x.outputs] for x in results]
        if all(len(x) == 1 for x in ret):
            ret = [x[0] for x in ret]
        return ret


class VLLMGeneratorFromLora(VLLMGenerator):
    def __init__(
            self,
            base_model_name_or_path: str,
            lora_weight_name_or_path: str,
            model_init_kwargs: Optional[dict] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            sampling_params_kwargs: Optional[dict] = None,
            generate_kwargs: Optional[dict] = None,
            json_schema: Optional[Type[BaseModel]] = None
    ):
        """
        Same as the VLLMGenerator, but support lora weight.

        :param base_model_name_or_path:
        :param lora_weight_name_or_path:
        :param model_init_kwargs: Additional kwargs for `vllm.LLM.__init__`.
        :param tokenizer: Tokenizer, will load a new tokenizer from the model_name_or_path if not provided.
        :param sampling_params_kwargs: Kwargs for `vllm.SamplingParams`.
        :param generate_kwargs: Kwargs for `vllm.LLM.generate`.
        :param json_schema: Json schema for lm-format-enforcer.
        """

        self.temp_dir_object = tempfile.TemporaryDirectory()
        temp_dir = self.temp_dir_object.name

        logger.debug("Preparing LoRA model weight for vLLM")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
        model_to_merge = PeftModel.from_pretrained(base_model, lora_weight_name_or_path)
        model = model_to_merge.merge_and_unload()
        model.save_pretrained(temp_dir)

        logger.debug("Preparing LoRA model tokenizer for vLLM")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(lora_weight_name_or_path)
        tokenizer.save_pretrained(temp_dir)

        super().__init__(
            model_name_or_path=temp_dir,
            tokenizer=tokenizer,
            model_init_kwargs=model_init_kwargs,
            sampling_params_kwargs=sampling_params_kwargs,
            generate_kwargs=generate_kwargs,
            json_schema=json_schema
        )

    def __del__(self):
        self.temp_dir_object.cleanup()
