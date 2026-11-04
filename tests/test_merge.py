"""The merge stage on fabricated views: two GVHMR-shaped outputs of one true motion, each in its own world frame,
one of them damaged where its keypoints are weak and absent for a stretch. The stage must put the views into one
frame (recovering the injected yaw and offset), prefer the sound view, keep the heading-relative root free of yaw,
weight the damaged stretch down, and write every documented field."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.fusion import geodesic, heading_yaw_deg   # noqa: E402
from wgvp.merge import merge_recording             # noqa: E402

BOX = {"left": [0, 0, 500, 600], "right": [500, 0, 1400, 500]}


def _view(d: Path, name, frames, body, root, transl, betas, conf, visible):
    d.mkdir(parents=True); T = len(frames); fps = 30.0
    np.savez(d / "smpl_global.npz", frame_idx=frames, pts_ms=frames / fps * 1000, imu_t_s=frames / fps - 2.0, body_pose=body.reshape(T, 63).astype(np.float32),
             global_orient=root.astype(np.float32), transl=transl.astype(np.float32), betas=np.tile(betas, (T, 1)).astype(np.float32), visible=visible,
             edge_truncated=np.zeros(T, bool), chunk_id=np.zeros(T, np.int16), pose_step_deg=np.zeros(T, np.float32), fps_source=fps, frame_step=1)
    x0, y0, x1, y1 = BOX[name]; kp = np.zeros((T, 17, 3), np.float32)
    kp[:, :, 0] = 0.5 * (x0 + x1); kp[:, :, 1] = 0.5 * (y0 + y1); kp[:, :, 2] = conf[:, None]
    np.savez(d / "kp2d_vitpose.npz", kp2d=kp, bbox_xyxy=np.zeros((T, 4), np.float32), view_box=np.array(BOX[name]), frame_idx=frames)


def _truth(T, rng):
    t = np.arange(T) / 30.0
    body = np.zeros((T, 21, 3)); body[:, 3, 0] = 0.5 + 0.4 * np.sin(2 * np.pi * 0.9 * t); body[:, 4, 0] = 0.5 - 0.4 * np.sin(2 * np.pi * 0.9 * t)   # knees
    body += np.cumsum(rng.normal(0, 0.004, (T, 21, 3)), axis=0)
    heading = 20.0 + 15.0 * np.sin(2 * np.pi * 0.05 * t)                                                     # a slow turn
    root = np.stack([(R.from_euler("y", h, degrees=True) * R.from_euler("x", 6, degrees=True)).as_rotvec() for h in heading])
    fwd = R.from_rotvec(root).apply([0.0, 0.0, 1.0]); transl = np.cumsum(1.2 * fwd / 30.0, axis=0) + [0.0, 0.95, 0.0]   # 1.2 m/s forward
    return body, root, transl, heading


def _noisy(body, sd, rng):
    n = R.from_rotvec(rng.normal(0, sd, (body.shape[0] * 21, 3))).as_rotvec().reshape(body.shape)
    return (R.from_rotvec(n.reshape(-1, 3)) * R.from_rotvec(body.reshape(-1, 3))).as_rotvec().reshape(body.shape)


def test_merge_two_views(tmp_path):
    rng = np.random.default_rng(1); T = 360; frames = np.arange(60, 60 + T)
    body, root, transl, heading = _truth(T, rng); betas = np.array([0.5, -0.3, 0.1, 0, 0, 0, 0, 0, 0, 0.0])
    # left: its own world (yaw -40, origin shifted), damaged and low-confidence over frames 120-200, damaged but
    # CONFIDENT over 220-250 (the case the weights must expose), absent over 260-300
    yaw_l, off_l = -40.0, np.array([1.5, 0.0, -2.0]); Rl = R.from_euler("y", yaw_l, degrees=True)
    body_l = _noisy(body, 0.03, rng); body_l[120:200] = _noisy(body[120:200], 0.5, rng); body_l[220:250] = _noisy(body[220:250], 0.5, rng)
    root_l = (Rl * R.from_rotvec(root)).as_rotvec(); transl_l = Rl.apply(transl) + off_l
    conf_l = np.full(T, 0.9); conf_l[120:200] = 0.1
    keep_l = np.ones(T, bool); keep_l[260:300] = False
    # right: the reference frame itself, mildly noisy, full confidence throughout
    body_r = _noisy(body, 0.03, rng); conf_r = np.full(T, 0.9)
    rec = tmp_path / "deliver" / "s_selfpace"
    _view(rec / "left", "left", frames[keep_l], body_l[keep_l], root_l[keep_l], transl_l[keep_l], betas + 0.2, conf_l[keep_l], np.ones(keep_l.sum(), bool))
    _view(rec / "right", "right", frames, body_r, root, transl, betas, conf_r, np.ones(T, bool))
    s = merge_recording(rec, log=lambda *a: None)
    assert s["status"] in ("PASS", "WARN") and (rec / "merged_smpl.npz").exists() and (rec / "merge.json").exists()
    m = np.load(rec / "merged_smpl.npz")
    for k in ("frame_idx", "imu_t_s", "body_pose", "global_orient", "transl", "heading_yaw_deg", "heading_yaw_rate_deg_s", "global_orient_heading_rel", "root_vel_heading_rel_m_s",
              "betas", "export_ok", "weight", "root_weight", "source", "usable", "share", "views", "reference_view", "view_alignment_yaw_deg", "view_alignment_offset_m", "joints", "fps", "fields_doc"):
        assert k in m.files, k
    assert len(m["frame_idx"]) == T and str(m["reference_view"]) == "right"                      # the view that sees more is the reference
    # the left view was turned onto the right view's world: the injected yaw and origin shift come back
    al = s["per_view"]["left"]["alignment"]
    assert abs(al["yaw_deg"] - 40.0) < 1.5 and np.allclose(al["offset_m"], -R.from_euler("y", 40.0, degrees=True).apply(off_l), atol=0.05)
    assert al["root_residual_deg"] < 3 and al["transl_residual_m"] < 0.05
    # merged body: close to the truth everywhere, and on the damaged stretch far closer than the damaged view
    err = geodesic(m["body_pose"].reshape(T, 21, 3), body); err_l = geodesic(body_l, body)
    assert np.nanmedian(err) < 3.0 and np.nanmedian(err[120:200]) < 0.3 * np.nanmedian(err_l[120:200])
    assert m["export_ok"].mean() > 0.95 and np.isfinite(m["body_pose"]).all()                     # the right view sees everything: nothing is dropped
    # root and translation: in the reference world, within a degree / a few cm of the truth
    assert np.nanmedian(geodesic(m["global_orient"], root)) < 2.0 and np.nanmedian(np.linalg.norm(m["transl"] - transl, axis=1)) < 0.05
    # heading-relative root: no yaw left; the tilt is the truth's 6 deg about x; the heading tracks the true heading
    fwd = R.from_rotvec(m["global_orient_heading_rel"]).apply([0.0, 0.0, 1.0])
    assert np.abs(np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]))).max() < 1e-3
    tilt = R.from_rotvec(m["global_orient_heading_rel"]).as_euler("YXZ", degrees=True)
    assert abs(np.median(tilt[:, 1]) - 6.0) < 1.0
    assert np.nanmedian(np.abs(((m["heading_yaw_deg"] - heading + 180) % 360) - 180)) < 2.0
    # velocity in the heading frame: forward 1.2 m/s, lateral ~0
    v = m["root_vel_heading_rel_m_s"]; ok = np.isfinite(v).all(1)
    assert abs(np.median(v[ok, 2]) - 1.2) < 0.1 and abs(np.median(v[ok, 0])) < 0.1
    # shape: one vector per recording, between the two views' estimates
    assert np.allclose(m["betas"], betas + 0.1, atol=0.11)
    # weights: in [0, 1], zero where no view saw the joint; one confident view (120-200, the left view excluded by its
    # keypoints) and two agreeing views (20-100) both carry ~1; a confident but wrong view averaged in (220-250) is exposed
    w = m["weight"]; assert w.min() >= 0 and w.max() <= 1 and (w[~np.isfinite(m["body_pose"].reshape(T, 21, 3)).all(2)] == 0).all()
    assert w[20:100].mean() > 0.9 and w[125:195].mean() > 0.95 and w[225:245].mean() < 0.5
    assert np.nanmedian(err[225:245]) > 3 * np.nanmedian(err[20:100])      # (and the merge there is indeed worse: the weight is what protects a learner)
    assert "transl_weight" in m.files and "transl_disagree_m" in m.files and np.nanmedian(m["transl_disagree_m"]) < 0.05 and m["transl_weight"][20:100].mean() > 0.95
    # source bits: left absent over 260-300 -> only the right view (bit 1) there; both elsewhere
    assert (m["source"][270:290] == 2).all() and (m["source"][20:100] == 3).all()
    assert m["root_weight"].min() >= 0 and m["root_weight"].max() <= 1 and json.loads((rec / "merge.json").read_text())["export"]["n_frames"] == int(m["export_ok"].sum())


def test_merge_single_view_and_gaps(tmp_path):
    """One lifted view: the merge is that view with its own usable mask, gaps where its legs are out of the panel
    become dropped frames, and short runs are not exported."""
    rng = np.random.default_rng(2); T = 300; frames = np.arange(T)
    body, root, transl, _ = _truth(T, rng); betas = np.zeros(10)
    conf = np.full(T, 0.9); rec = tmp_path / "deliver" / "r"
    _view(rec / "right", "right", frames, body, root, transl, betas, conf, np.ones(T, bool))
    k = np.load(rec / "right" / "kp2d_vitpose.npz"); kp = k["kp2d"].copy()
    kp[100:160, 13:17, 1] = 10_000.0                           # knees and ankles out of the panel for 2 s
    kp[200:208, 13:17, 1] = 10_000.0                           # and for 8 frames: the runs either side stay, the gap is a gap
    np.savez(rec / "right" / "kp2d_vitpose.npz", kp2d=kp, bbox_xyxy=k["bbox_xyxy"], view_box=k["view_box"], frame_idx=frames)
    s = merge_recording(rec, log=lambda *a: None); m = np.load(rec / "merged_smpl.npz")
    assert s["views"] == ["right"] and str(m["reference_view"]) == "right" and m["share"].shape == (1, T) and np.allclose(m["share"][0], 1.0)
    bp = m["body_pose"].reshape(T, 21, 3)
    assert np.isnan(bp[100:160, 3]).all() and np.isfinite(bp[:100, 3]).all()               # knees not output where not seen
    assert not m["export_ok"][100:160].any() and not m["export_ok"][200:208].any() and m["export_ok"][:100].all() and m["export_ok"][208:].all()
    assert (m["weight"][100:160] == 0).all() and (m["source"][100:160, 3] == 0).all() and (m["source"][:100, 3] == 1).all()
    assert np.nanmedian(geodesic(bp, body)) < 1.0 and abs(float(m["view_alignment_yaw_deg"][0])) == 0.0
