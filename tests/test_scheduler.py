import sys
import os
from pathlib import Path
import json
import yaml
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_orchestrator_cli(tmp_path):
    # create dummy orchestrator module
    mod = tmp_path / "dummy_mod.py"
    mod.write_text("def dummy(prompt):\n    return {'echo': prompt}\n")

    cfg = {
        "orchestrators": {
            "dummy": {"module": mod.stem, "callable": "dummy"}
        },
        "default_orchestrator": "dummy",
    }
    cfg_file = tmp_path / "zsi_config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    cmd = [sys.executable, "-m", "core.orchestrator", "--strategy", "dummy", "--prompt", "hi", "-c", str(cfg_file), "--json"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), env.get("PYTHONPATH", "")])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out == {"echo": "hi"}
