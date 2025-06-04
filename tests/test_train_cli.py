import os
import subprocess
import sys
from pathlib import Path


def test_train_mamba_parses_model_config():
    root = Path(__file__).resolve().parents[1]
    script = root / "src" / "train_mamba.py"
    config = root / "configs" / "mamba_config.json"
    env = os.environ.copy()
    stubs = root / "tests" / "stubs"
    env["PYTHONPATH"] = f"{stubs}{os.pathsep}" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(script), "--model-config-path", str(config), "--help"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0
    assert "model-config-path" in result.stdout
    assert "dataset-path" in result.stdout
