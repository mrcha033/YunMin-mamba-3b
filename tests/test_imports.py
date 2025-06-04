import py_compile
from pathlib import Path

# Ensure the training script and generated modules are syntactically valid

def test_compile_train_mamba():
    py_compile.compile(str(Path(__file__).resolve().parents[1] / "train_mamba.py"), doraise=True)


def test_compile_sagemaker_launcher():
    """Ensure the SageMaker launcher script is syntactically valid"""
    py_compile.compile(
        str(Path(__file__).resolve().parents[1] / "sagemaker_training_job.py"),
        doraise=True,
    )

