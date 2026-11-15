"""The demo video, rendered from the bundled example with the stand-in body model (tools/standin_body_model.py): the
renderer opens an offscreen OpenGL context on this machine, the file decodes to the frames it claims, the frames carry
ink (meshes, filmstrip, charts) and change over time.

It needs the demo extras (pip install -e .[demo]) and a display or a software OpenGL; without them it is skipped,
except under WGVP_DEMO=1 (the CI sets it on Windows, macOS and Linux), where it must run."""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
MUST = os.environ.get("WGVP_DEMO") == "1"


def _skip_or_fail(reason: str):
    if MUST:
        pytest.fail(f"WGVP_DEMO=1 and the demo cannot run here: {reason}")
    pytest.skip(reason)


def test_the_demo_renders_with_the_standin_body(tmp_path):
    from wgvp.demo import extras_missing, render_demo
    miss = extras_missing()
    if miss:
        _skip_or_fail(f"demo extras missing: {', '.join(miss)}")
    sys.path.insert(0, str(REPO / "tools"))
    import standin_body_model
    bm = standin_body_model.write(tmp_path / "body_models").parents[1]
    root = tmp_path / "data root été"
    assert (REPO / "examples/bundled/wgvp_output/OWN01/deliver/OWN01_selfpace/merged_smpl.npz").exists(), \
        "examples/bundled has been run in place and lost its lifted outputs: git checkout -- examples/bundled"
    shutil.copytree(REPO / "examples" / "bundled", root)                  # the inputs and the lifted outputs of OWN01 / OWN02
    lines = []
    try:
        res = render_demo(root, "OWN01", recording="OWN01_selfpace", body_models=str(bm), step=4, log=lines.append)
    except RuntimeError as e:
        if "OpenGL" in str(e) or "display" in str(e):
            _skip_or_fail(str(e))
        raise
    out = Path(res["video"])
    assert out.exists() and out.parent == root / "wgvp_output" / "OWN01" / "work" / "demo", res
    assert res["n_frames"] == 99 and res["session"] == "OWN01", res          # 393 lifted frames, every fourth
    import av
    frames = []
    with av.open(str(out)) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    assert len(frames) == res["n_frames"], (len(frames), res)
    assert list(frames[0].shape[:2]) == [res["size"][1], res["size"][0]], (frames[0].shape, res["size"])
    ink = [float(((g > 40) & (g < 215)).mean()) for g in (f.mean(2) for f in frames[::10])]   # neither background nor white: meshes, strip, charts
    assert all(0.05 < v < 0.6 for v in ink), ink
    moved = float((np.abs(frames[0].astype(int) - frames[-1].astype(int)).max(2) > 20).mean())
    assert moved > 0.01, moved                                               # the walker, the cursor and the readout move
    assert any(line.startswith("demo -> ") for line in lines), lines[-3:]
