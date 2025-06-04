import py_compile
from pathlib import Path

# Ensure the training script and generated modules are syntactically valid

def test_compile_train_mamba():
    py_compile.compile(str(Path(__file__).resolve().parents[1] / "train_mamba.py"), doraise=True)


def test_mamba_lmheadmodel_instantiation():
    import json
    from transformers import MambaLMHeadModel, MambaConfig
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

