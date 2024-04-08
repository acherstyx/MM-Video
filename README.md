# MM-Video

> [!CAUTION]
> This project is still in progress, the code is unstable and keeps changing. Use it at your own risk.

## Basic Usage

To use this codebase, you can either integrate it as a dependency into your project or directly fork this repository.
We recommend the former option.

### Step 1. Add Dependency

**pyproject.toml**:

```toml
...
dependencies = [
"mm-video@git+https://github.com/acherstyx/MM-Video.git@develop",
...
]
...
```

**requirements.txt**:

```txt
git+https://github.com/acherstyx/MM-Video.git@develop
```

### Step 2. Build Experiment

Create your own Hydra config and add `mm_video` to the default list.

**configs/config.yaml**:

```yaml
defaults:
  - mm_video
```

**main.py**:

```python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    runner = instantiate(cfg.runner)
    runner.run(cfg)


if __name__ == "__main__":
    main()
```

### Step 3. Customization and Registration

A default trainer and runner are defined in the init.   
You can define your own dataset, model, meter, trainer, and runner, and register them into the config group. 
For example:

```python
from torch.utils.data import Dataset
from torch.nn import Module

from mm_video.config import dataset_store, model_store


@dataset_store()
class MyDataset(Dataset):
    def __init__(self, data_root: str):
        ...


@model_store()
class MyModel(Module):
    def __init__(self, n_layers: int):
        ...
```
