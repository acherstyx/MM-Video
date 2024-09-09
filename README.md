# MM-Video

`mm-video` serve as the basic template to help you build your deep learning project quickly.

**🌟Features:**

- 😌 Easy to use: **NO NEED to clone or fork**—just add it as a dependency and start your project.
- 📝 Experiment configuration management powered by Hydra.
- 🌈 Colorful logging.
- 🛠️ Lots of utils (Contributions and suggestions are welcome! I'm actively working on it.)

Before using this codebase, it's important to be familiar with Hydra.
Learn more at the [Hydra Documentation](https://hydra.cc/docs/intro/).

## Get Start

To start using it, simply add it to your dependencies.

**pyproject.toml**:

```toml
dependencies = [
"mm-video@git+https://github.com/acherstyx/MM-Video.git@master",
]
```

**requirements.txt**:

```txt
git+https://github.com/acherstyx/MM-Video.git@master
```

**NOTE**: For stability, we recommend using the `master` branch.
The `develop` branch is periodically updated and may contain untested code.
If needed, you can specify a specific version (this is recommended).
More examples can be found here:

```bash
# Tag
git+https://github.com/acherstyx/MM-Video.git@v0.4.0
# Commit hash
git+https://github.com/acherstyx/MM-Video.git@1ff78ee
# With optional dependencies
mm-video[utils_common,utils_language,tools]@git+https://github.com/acherstyx/MM-Video@master
```

## Basic Usage

Coming soon.
