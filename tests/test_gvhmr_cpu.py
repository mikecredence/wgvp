"""The lift without a GPU: GVHMR's script runs unchanged, with its moves to the GPU switched off."""
import os
import subprocess
import sys
from pathlib import Path

from wgvp import gvhmr_stage

REPO = Path(__file__).resolve().parents[1]

SCRIPT = """
import sys, torch
x = torch.zeros(3).cuda(); m = torch.nn.Linear(3, 2).cuda()
torch.cuda.synchronize(); torch.cuda.empty_cache()
torch.save({"w": m.weight.detach()}, sys.argv[1]); back = torch.load(sys.argv[1])
print("device", x.device.type, m.weight.device.type, back["w"].device.type, "|", torch.cuda.get_device_name(), "| args", sys.argv[1:])
"""


def test_a_script_that_calls_cuda_runs_on_a_machine_without_one(tmp_path):
    script = tmp_path / "demo_like.py"; script.write_text(SCRIPT, encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, "-m", "wgvp.gvhmr_cpu", str(script), str(tmp_path / "w.pt"), "-s"], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-800:]
    assert "device cpu cpu cpu | none (lifting on the CPU)" in p.stdout and "-s" in p.stdout, p.stdout


def test_the_stage_runs_the_demo_through_the_wrapper_only_when_asked(tmp_path, monkeypatch):
    groot = tmp_path / "GVHMR"; (groot / "tools/demo").mkdir(parents=True)
    (groot / "tools/demo/demo.py").write_text("# GVHMR_RENDER guard\n", encoding="utf-8")
    seen = []

    def fake_run(cmd, **kw):
        seen.append((cmd, kw["env"]))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(gvhmr_stage.subprocess, "run", fake_run)
    for device in ("cuda", "cpu"):
        gvhmr_stage.run_demo(groot, tmp_path / "v.mp4", tmp_path / "pred", 10, tmp_path / "run.log", f_mm=26.2, device=device)
    (gpu_cmd, gpu_env), (cpu_cmd, cpu_env) = seen
    assert gpu_cmd[1].endswith("demo.py") and "wgvp.gvhmr_cpu" not in gpu_cmd and gpu_env.get("CUDA_VISIBLE_DEVICES") != ""
    assert cpu_cmd[1:3] == ["-m", "wgvp.gvhmr_cpu"] and cpu_cmd[3].endswith("demo.py") and cpu_cmd[-2:] == ["--f_mm", "26"]
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == "" and str(REPO) in cpu_env["PYTHONPATH"]
