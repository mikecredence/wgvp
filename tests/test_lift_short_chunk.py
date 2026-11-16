"""tools/lift_short_chunk.py: the two-second chunk it cuts keeps the chunk's own frames and boxes, and its check against
the truth passes a lift that is close and fails one that is not (the lift itself needs GVHMR and runs in CI's
lift-cpu workflow)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import lift_short_chunk as lsc  # noqa: E402


def _chunk(root: Path, n: int = 24) -> Path:
    import cv2
    d = lsc.chunk_dir(root, "left"); d.mkdir(parents=True)
    w = cv2.VideoWriter(str(d / "chunk_0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(n):
        f = np.zeros((48, 64, 3), np.uint8); f[:, : 2 + 2 * i] = 255; w.write(f)
    w.release()
    np.save(d / "chunk_0_bbox.npy", np.tile(np.array([[1.0, 2.0, 30.0, 40.0]], np.float32), (n, 1)) + np.arange(n, dtype=np.float32)[:, None])
    (d / "chunk_0.json").write_text(json.dumps({"chunk_id": 0, "video": "chunk_0.mp4", "bbox": "chunk_0_bbox.npy", "n_frames": n,
                                                 "frame_idx": list(range(100, 100 + n)), "focal_35mm": 27.0, "status": "done", "pose_npz": "x"}), encoding="utf-8")
    return d


def test_prepare_keeps_the_first_frames_and_their_boxes(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import cv2
    _chunk(tmp_path / "src"); d = lsc.prepare(tmp_path / "src", tmp_path / "short", frames=10, view="left")
    m = json.loads((d / "chunk_0.json").read_text(encoding="utf-8"))
    assert m["n_frames"] == 10 and m["frame_idx"] == list(range(100, 110)) and m["status"] == "pending" and m["focal_35mm"] == 27.0 and "pose_npz" not in m
    assert np.load(d / "chunk_0_bbox.npy").shape == (10, 4) and float(np.load(d / "chunk_0_bbox.npy")[9, 0]) == 10.0
    cap = cv2.VideoCapture(str(d / "chunk_0.mp4")); n = 0
    while cap.read()[0]:
        n += 1
    assert n == 10


def test_check_passes_a_close_lift_and_fails_a_wrong_one(tmp_path, capsys):
    gt = REPO / "sample" / "HC900_gt.npz"; G = np.load(gt); n = 30
    d = lsc.chunk_dir(tmp_path, "left"); d.mkdir(parents=True)
    frames = [int(f) for f in G["frame_idx"][40:40 + n]]; gi = {int(f): i for i, f in enumerate(G["frame_idx"])}
    body = G["pose_aa"][[gi[f] for f in frames], 3:66].astype(np.float32)
    rng = np.random.default_rng(0)
    for name, noise, expect in (("close", 0.05, 0), ("wrong", 1.5, 1)):
        np.savez(d / f"{name}_pose.npz", incam_body_pose=body + rng.normal(0, noise, body.shape).astype(np.float32), global_body_pose=body,
                 incam_global_orient=np.zeros((n, 3), np.float32), global_transl=np.zeros((n, 3), np.float32), incam_betas=np.zeros((n, 10), np.float32))
        (d / "chunk_0.json").write_text(json.dumps({"n_frames": n, "frame_idx": frames, "status": "done", "pose_npz": f"{name}_pose.npz",
                                                     "gvhmr_run": {"elapsed_s": 90.0}}), encoding="utf-8")
        assert lsc.check(tmp_path, gt, 20.0, "left") == expect, (name, capsys.readouterr().out)
    (d / "chunk_0.json").write_text(json.dumps({"n_frames": n, "status": "failed", "error": "boom"}), encoding="utf-8")
    assert lsc.check(tmp_path, gt, 20.0, "left") == 1
