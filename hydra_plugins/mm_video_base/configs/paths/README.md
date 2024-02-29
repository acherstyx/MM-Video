# Paths

This directory is currently unused, but it can serve as a central location for storing paths within the unified `path`
folder, enhancing management convenience. The stored paths can be integrated into specific configurations using
OmegaConf's [interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) syntax.

## Example

Suppose you are running your experiment on different environments (Machine A and Machine B); you can store your training
data in different locations and set the path for each machine separately.

`configs/paths/machine_a.yaml`:

```yaml
image_dataset_root: "/path/to/image/root/on/machine_a"
```

`configs/paths/machine_b.yaml`:

```yaml
image_dataset_root: "/path/to/image/root/on/machine_b"
```

`configs/experiments/exp.yaml`:

```yaml
# @package _global_

data_loader:
  dataset:
    video_root: ${paths.image_dataset_root}
``` 

Then you can run your experiment on different machines with override settings:

- On machine A: `torchrun -m xxx.xxx.xxx +experiments=exp +paths=machine_a`
- On machine B: `torchrun -m xxx.xxx.xxx +experiments=exp +paths=machine_b`
