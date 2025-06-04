# Contributing

Thank you for your interest in contributing! Before running the test suite, make sure the required packages are installed.

## Installing Dependencies for Tests

The unit tests rely on Hugging Face's `transformers` library and the `mamba-ssm` package. Install them with pip:

```bash
pip install transformers mamba-ssm
```

You will also need `pytest` and the libraries listed in `requirements.txt` to run the full test suite.

## Running Tests

After installing the dependencies, execute:

```bash
pytest
```
