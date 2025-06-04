# Contributing

Thank you for your interest in contributing to this project!

## Running Tests

The unit tests require a few optional libraries that are not installed by default.
Before executing `pytest`, install `transformers` and `mamba-ssm`:

```bash
pip install transformers mamba-ssm
```

You will also need `pytest` and the libraries listed in `requirements.txt` to run the full test suite. If `nvcc` is not available on your
system, set `MAMBA_SKIP_CUDA_BUILD=1` when installing these dependencies so the CUDA kernels are skipped:

```bash
MAMBA_SKIP_CUDA_BUILD=1 pip install -r requirements.txt
```

## Running Tests

After installing the dependencies, execute:

```bash
pytest
```
