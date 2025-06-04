import py_compile
from pathlib import Path

# Ensure the training script and generated modules are syntactically valid

def test_compile_train_mamba():
    py_compile.compile(
        str(Path(__file__).resolve().parents[1] / "src" / "train_mamba.py"),
        doraise=True,
    )


def test_mamba_lmheadmodel_instantiation():
    """Instantiate ``MambaLMHeadModel`` from the provided config."""
    import json
    import pytest

    try:
        from transformers import MambaLMHeadModel, MambaConfig
    except Exception as exc:  # pragma: no cover - skip if unavailable
        pytest.skip(f"MambaLMHeadModel unavailable: {exc}")

    # Use a very small configuration to avoid excessive memory usage during CI
    cfg = {
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
    }
    model = MambaLMHeadModel(MambaConfig(**cfg))
    assert model is not None

def test_compile_sagemaker_launcher():
    """Ensure the SageMaker launcher script is syntactically valid"""
    py_compile.compile(
        str(Path(__file__).resolve().parents[1] / "sagemaker" / "sagemaker_training_job.py"),
        doraise=True,
    )


def test_compile_sagemaker_spot_launcher():
    """Ensure the SageMaker Spot launcher script is syntactically valid"""
    py_compile.compile(
        str(Path(__file__).resolve().parents[1] / "sagemaker" / "sagemaker_spot_training_job.py"),
        doraise=True,
    )


def test_parse_dockerfile():
    """Basic syntax check for the Dockerfile using dockerfile-parse"""
    import pytest
    try:
        from dockerfile_parse import DockerfileParser
    except ImportError as exc:  # pragma: no cover - skip if unavailable
        pytest.skip(f"dockerfile_parse unavailable: {exc}")
    dockerfile_path = Path(__file__).resolve().parents[1] / "docker" / "Dockerfile"
    parser = DockerfileParser(str(dockerfile_path))
    assert parser.baseimage is not None

