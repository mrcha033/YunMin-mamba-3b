# Contributing

Thank you for your interest in contributing to this project!

## Running Tests

The unit tests require a few optional libraries that are not installed by default.
Before executing `pytest`, install `transformers`, `mamba-ssm`, and `dockerfile-parse`:

```bash
pip install transformers mamba-ssm dockerfile-parse
```

You will also need `pytest` and the libraries listed in `requirements.txt` to run the full test suite.  The easiest
way is to install everything at once:

```bash
MAMBA_SKIP_CUDA_BUILD=1 pip install -r requirements.txt
```

After installing the dependencies, execute:

```bash
pytest
```
