"""The stop: a run ends at the next check, keeps what finished, marks the session under way, and the lift's process
tree (the stage and the lifter it starts) ends with it."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# a stand-in for the GPU stage: it reports two pieces, starts a long-lived child the way the stage starts the
# lifter, records both process ids, and waits
FAKE_STAGE = """
import os, subprocess, sys
print("[gvhmr_stage] 2 chunk(s)", flush=True)
print("[gvhmr_stage] run  OWN01_selfpace/left/chunk_000 (10 frames)", flush=True)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
open(os.environ["WGVP_TEST_PIDFILE"], "w").write(f"{os.getpid()} {child.pid}")
child.wait()
print("[gvhmr_stage]      -> done  (0s)", flush=True)
"""


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")                       # a zombie answers kill(0) but is not running


def test_check_stop():
    from wgvp.progress import RunStopped, check_stop
    check_stop(None); check_stop(lambda: False)
    with pytest.raises(RunStopped):
        check_stop(lambda: True)


def test_stop_during_tracking_keeps_what_finished_and_marks_the_session(tmp_path):
    from wgvp.config import load_config
    from wgvp.session import run_root
    root = tmp_path / "root"
    shutil.copytree(REPO / "examples" / "bundled" / "OWN01", root / "OWN01")
    shutil.copy(REPO / "examples" / "bundled" / "wgvp_config.json", root)
    seen = {"frames": 0, "asked": None}

    def progress(stage, i, n, msg):
        if stage == "frames":
            seen["frames"] = max(seen["frames"], i)

    def stop():                                                    # the Stop button, pressed 30 frames into the tracking
        if seen["asked"] is None and seen["frames"] >= 30:
            seen["asked"] = time.time()
        return seen["asked"] is not None

    lines = []
    summ = run_root(root, load_config(root), cpu_only=True, log=lines.append, progress=progress, stop=stop)
    assert seen["asked"] is not None and time.time() - seen["asked"] < 30, "the run did not end soon after the stop"
    assert seen["frames"] < 100, seen                                 # the clip has 393 frames; the tracking did not go on
    assert summ["status"] == "STOPPED" and summ["stopped"] == {"after": 0, "of": 1, "during": "OWN01"}
    ss = json.loads((root / "wgvp_output" / "OWN01" / "deliver" / "session_summary.json").read_text(encoding="utf-8"))
    assert ss["status"] == "STOPPED" and "stopped by the user" in ss["error"]
    assert (root / "wgvp_output" / "QC_SUMMARY.html").exists()      # the summary still covers what there is
    assert any("stopped by the user: 0 of 1 session(s) finished, OWN01 not finished" in ln for ln in lines), lines[-3:]


def test_stop_ends_the_lift_stage_and_the_lifter_it_started(tmp_path, monkeypatch):
    from wgvp import session
    from wgvp.progress import RunStopped
    fake = tmp_path / "fake_stage.py"; fake.write_text(FAKE_STAGE, encoding="utf-8")
    if os.name == "nt":                                              # "the GVHMR environment's python": a wrapper that runs the stand-in
        gpy = tmp_path / "python.bat"; gpy.write_text(f'@"{sys.executable}" "{fake}"\r\n', encoding="utf-8")
    else:
        gpy = tmp_path / "python"; gpy.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}"\n', encoding="utf-8"); gpy.chmod(0o755)
    pidfile = tmp_path / "pids.txt"; monkeypatch.setenv("WGVP_TEST_PIDFILE", str(pidfile))
    root = tmp_path / "root"; root.mkdir()
    cfg = {"gvhmr": {"root": str(tmp_path), "timeout_s": 60}}
    lifts = []
    t0 = time.time()
    with pytest.raises(RunStopped):
        session.run_gvhmr_stage(root, cfg, gvhmr_python=str(gpy), log=lambda m: None, progress=lambda *a: lifts.append(a), stop=pidfile.exists)
    assert time.time() - t0 < 20
    parent, child = [int(x) for x in pidfile.read_text(encoding="utf-8").split()]
    for _ in range(40):
        if not _alive(parent) and not _alive(child):
            break
        time.sleep(0.25)
    assert not _alive(child), "the lifter (the stage's child) is still running after the stop"
    assert not _alive(parent), "the stage is still running after the stop"
    assert ("lift", 0, 2, "OWN01_selfpace/left/chunk_000") in lifts  # the stage's progress came through before the stop
