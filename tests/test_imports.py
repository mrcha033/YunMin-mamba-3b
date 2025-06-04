import py_compile
from pathlib import Path

# Ensure the training script and generated modules are syntactically valid

def test_compile_train_mamba():
    py_compile.compile(str(Path(__file__).resolve().parents[1] / "train_mamba.py"), doraise=True)


def test_mamba_lmheadmodel_instantiation():
    import json
    import pytest
    try:
        from transformers import MambaLMHeadModel, MambaConfig
    except Exception:
        pytest.skip("transformers not available")
    cfg_path = Path(__file__).resolve().parents[1] / "mamba_config.json"
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

