# -*- coding: utf-8 -*-
# @Time    : 6/30/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_generate.py

from typing import List

import torch
from pytest import fixture, mark, param
from transformers import AutoTokenizer, PreTrainedTokenizer

from mm_video.utils.language.generate import *


@fixture()
def generator(generator_type: str, model_name: str) -> Generator:
    if generator_type == "hf":
        generator = HFGenerator(model_name_or_path=model_name, batch_size=2)
        yield generator
        # cleanup
        del generator.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    elif generator_type == "hf_torch_compile":
        generator = HFGenerator(model_name_or_path=model_name, batch_size=2, torch_compile=True)
        yield generator
        # cleanup
        del generator.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    elif generator_type == "vllm":
        yield VLLMGenerator(model_name_or_path=model_name)
    elif generator_type == "hf_pipeline":
        yield HFPipelineGenerator(
            model_name,
            pipeline_kwargs=dict(device="cuda"),
            generate_kwargs=dict(
                max_new_tokens=64, temperature=1.0, do_sample=True, top_k=50
            ),
            batch_size=2
        )
    else:
        raise NotImplementedError


@fixture()
def tokenizer(model_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


@fixture()
def prompts() -> List[str]:
    return ["hello!", "print hello world in python", "A story about rabbit:", "A story about cat:"]


def print_prompt_and_response(prompts: List[str], responses: List[str]) -> None:
    for p, r in zip(prompts, responses):
        print(f"Prompt:\n   {p}")
        print(f"Response:\n   {r}")
        print("-" * 50)


def print_prompt_and_multiple_responses(prompts: List[str], responses: List[List[str]]) -> None:
    for p, rs in zip(prompts, responses):
        print(f"Prompt:\n\t{p}")
        print("Response:")
        for i, r in enumerate(rs):
            print(f"[{i}]: {r}")
        print("-" * 50)


class GeneratorTestSuit:
    # noinspection PyMethodMayBeStatic
    def test_generate(self, prompts: List[str], generator: Generator) -> None:
        responses = generator.generate(prompts)
        assert len(prompts) == len(responses)
        for resp in responses:
            assert type(resp) is str
        print_prompt_and_response(prompts, responses)

    def test_generate_with_chat_template(
            self,
            prompts: List[str], generator: Generator, tokenizer: PreTrainedTokenizer
    ) -> None:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        self.test_generate(prompts=prompts, generator=generator)


@mark.parametrize(
    "generator_type, model_name",
    [param("hf", "mistralai/Mistral-7B-Instruct-v0.3", id="HFGenerator"),
     param("hf_torch_compile", "mistralai/Mistral-7B-Instruct-v0.3", id="HFGenerator")]
)
class TestHFGenerator(GeneratorTestSuit):
    pass


@mark.parametrize(
    "generator_type, model_name",
    [param("hf_pipeline", "mistralai/Mistral-7B-Instruct-v0.3", id="HFPipelineGenerator")]
)
class TestHFPipelineGenerator(GeneratorTestSuit):
    def test_generate_beam_search(self, prompts, generator: HFPipelineGenerator, n: int = 3) -> None:
        responses = generator.generate(prompts, num_beams=n, num_return_sequences=n)
        assert len(responses) == len(prompts)
        assert all(type(resp) is list for resp in responses)
        assert all(len(resp) == n for resp in responses)
        print_prompt_and_multiple_responses(prompts, responses)


@mark.parametrize(
    "generator_type, model_name",
    [param("vllm", "mistralai/Mistral-7B-Instruct-v0.3", id="VLLMGenerator"), ]
)
class TestVLLMGenerator(GeneratorTestSuit):
    def test_generate_beam_search(self, prompts: List[str], generator: VLLMGenerator, n: int = 3) -> None:
        responses = generator.generate(prompts, n=n, best_of=n, use_beam_search=True, temperature=0)
        assert len(responses) == len(prompts)
        assert all(len(resp) == n for resp in responses)
        print_prompt_and_multiple_responses(prompts, responses)

    def test_generate_beam_search_multinomial(self, prompts: List[str], generator: VLLMGenerator, n: int = 3) -> None:
        responses = generator.generate(prompts, n=n, best_of=n, use_beam_search=False)
        assert len(responses) == len(prompts)
        assert all(len(resp) == n for resp in responses)
        print_prompt_and_multiple_responses(prompts, responses)
