"""Merging GVHMR chunk outputs: per-frame quality fields (visible, edge_truncated, pose_step_deg, chunk_id) and the
chunk-boundary semantics, on synthetic chunk files shaped like the GPU stage's."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.gvhmr_stage import assemble      # noqa: E402


def _chunk(work: Path, cid: int, frames: np.ndarray, boxes: np.ndarray, body_pose: np.ndarray, visible: np.ndarray, crop=(100, 0, 740, 480), scale=1.0):
    n = len(frames)
    np.savez_compressed(work / f"chunk_{cid}_pose.npz",
                        **{f"{fr}_{k}": v for fr in ("global", "incam") for k, v in
                           (("global_orient", np.zeros((n, 3), np.float32)), ("body_pose", body_pose.astype(np.float32)), ("betas", np.zeros((n, 10), np.float32)), ("transl", np.zeros((n, 3), np.float32)))},
                        K_fullimg=np.eye(3, dtype=np.float32), kp2d_vitpose=np.zeros((n, 17, 3), np.float32), bbx_xyxy=boxes.astype(np.float32), static_conf_logits=np.zeros((n, 6), np.float32))
    np.save(work / f"chunk_{cid}_bbox.npy", boxes.astype(np.float32))
    (work / f"chunk_{cid}.json").write_text(json.dumps({"chunk_id": cid, "video": f"chunk_{cid}.mp4", "bbox": f"chunk_{cid}_bbox.npy", "n_frames": n, "frame_idx": frames.tolist(),
                                                        "pts_ms": (frames / 30.0 * 1000).tolist(), "imu_t_s": (frames / 30.0 - 1.0).tolist(), "visible": visible.tolist(), "frame_step": 1,
                                                        "cut_scale": scale, "crop": list(crop), "fps_source": 30.0, "status": "done", "pose_npz": f"chunk_{cid}_pose.npz"}))


def test_assemble_quality_fields(tmp_path):
    work = tmp_path / "chunks"; work.mkdir(); deliver = tmp_path / "deliver"
    rng = np.random.default_rng(0)
    # chunk 0: frames 0-99, the box touches the top edge (head cut) in frames 40-59; frames 70-79 are bridged (not visible)
    f0 = np.arange(100); b0 = np.tile([200.0, 50.0, 300.0, 400.0], (100, 1)); b0[40:60, 1] = 0.0
    v0 = np.ones(100, bool); v0[70:80] = False
    p0 = np.cumsum(rng.normal(0, 0.01, (100, 63)), axis=0)               # smooth pose, small steps
    p0[90:] += 1.0                                                        # one deliberate, persistent jump at frame 90 (> 30 deg on every joint)
    # chunk 1: after a 3-s absence, frames 190-249, a different pose: the step across the boundary must be NaN, not a jump
    f1 = np.arange(190, 250); b1 = np.tile([150.0, 60.0, 260.0, 470.0], (60, 1)); b1[-5:, 3] = 480.0    # feet at the bottom edge at the end
    v1 = np.ones(60, bool); p1 = np.full((60, 63), 0.8) + np.cumsum(rng.normal(0, 0.01, (60, 63)), axis=0)
    _chunk(work, 0, f0, b0, p0, v0); _chunk(work, 1, f1, b1, p1, v1)
    r = assemble(work, deliver); assert r["ok"] and r["n_frames"] == 160 and r["n_chunks"] == 2
    z = np.load(deliver / "smpl_global.npz")
    for k in ("visible", "edge_truncated", "pose_step_deg", "chunk_id"):
        assert k in z.files and len(z[k]) == 160
    assert z["visible"].sum() == 150 and (~z["visible"][70:80]).all()
    et = z["edge_truncated"]; assert et[40:60].all() and not et[:40].any() and not et[60:100].any() and et[-5:].all()
    st = z["pose_step_deg"]; assert np.isnan(st[0]) and np.isnan(st[100]) and np.isfinite(st[1:100]).all()        # NaN at each chunk start only
    assert np.nanmax(st[1:90]) < 5 and st[90] > 30 and np.nanmax(st[91:100]) < 5                                  # the planted jump, nothing else
    assert (z["chunk_id"][:100] == 0).all() and (z["chunk_id"][100:] == 1).all()
    gm = json.loads((deliver / "gvhmr_meta.json").read_text())
    assert gm["n_bridged"] == 10 and gm["n_edge_truncated"] == 25 and gm["n_pose_jumps_over_30deg"] == 1 and gm["pose_jump_frames"] == [90]


def test_assemble_reports_missing_chunk_dumps_instead_of_crashing(tmp_path):
    """Chunk records pulled back from a GPU box say 'done' but the pose dumps stayed there: assembly must say so
    (and whether an assembled output already exists), not raise."""
    work = tmp_path / "chunks"; work.mkdir(); deliver = tmp_path / "deliver"; deliver.mkdir()
    f0 = np.arange(30); b0 = np.tile([200.0, 50.0, 300.0, 400.0], (30, 1))
    _chunk(work, 0, f0, b0, np.zeros((30, 63)), np.ones(30, bool)); (work / "chunk_0_pose.npz").unlink()
    r = assemble(work, deliver); assert r["ok"] is False and r["reason"] == "chunk pose file(s) missing" and r["already_assembled"] is False
    (deliver / "smpl_global.npz").write_bytes(b""); r = assemble(work, deliver); assert r["already_assembled"] is True
