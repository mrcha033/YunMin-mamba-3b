import py_compile
from pathlib import Path

# Ensure the training script and generated modules are syntactically valid

def test_compile_train_mamba():
    py_compile.compile(str(Path(__file__).resolve().parents[1] / "train_mamba.py"), doraise=True)


def test_mamba_lmheadmodel_instantiation():
    """Instantiate ``MambaLMHeadModel`` from the provided config."""
    import json
    import pytest

    try:
        from transformers import MambaLMHeadModel, MambaConfig
    except Exception as exc:  # pragma: no cover - skip if unavailable
        pytest.skip(f"MambaLMHeadModel unavailable: {exc}")

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "mamba_config.json"
    cfg = json.loads(cfg_path.read_text())
    model = MambaLMHeadModel(MambaConfig(**cfg))
    assert model is not None

def test_compile_sagemaker_launcher():
    """Ensure the SageMaker launcher script is syntactically valid"""
    py_compile.compile(
        str(Path(__file__).resolve().parents[1] / "sagemaker_training_job.py"),
        doraise=True,
    )


def test_parse_dockerfile():
    """Basic syntax check for the Dockerfile using dockerfile-parse"""
    from dockerfile_parse import DockerfileParser
    dockerfile_path = Path(__file__).resolve().parents[1] / "Dockerfile"
    parser = DockerfileParser(str(dockerfile_path))
    assert parser.baseimage is not None

