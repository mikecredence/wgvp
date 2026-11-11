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
from wgvp.merge import align_to_reference, merge_recording, moving_frames   # noqa: E402

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


def test_export_needs_legs_and_torso_but_not_arms(tmp_path):
    """The export rule: a frame whose torso no view saw is dropped whole, like a frame without legs; a frame with
    only an arm unseen is kept, the arm's joints NaN with weight 0."""
    rng = np.random.default_rng(5); T = 300; frames = np.arange(T)
    body, root, transl, _ = _truth(T, rng); rec = tmp_path / "deliver" / "r"
    _view(rec / "right", "right", frames, body, root, transl, np.zeros(10), np.full(T, 0.9), np.ones(T, bool))
    k = np.load(rec / "right" / "kp2d_vitpose.npz"); kp = k["kp2d"].copy()
    kp[60:120, [5, 6], 1] = 10_000.0                           # both shoulders out of the panel for 2 s: no spine, neck, head (and no arms)
    kp[180:240, [7, 9], 1] = 10_000.0                          # left elbow and wrist out for 2 s: the left arm only
    np.savez(rec / "right" / "kp2d_vitpose.npz", kp2d=kp, bbox_xyxy=k["bbox_xyxy"], view_box=k["view_box"], frame_idx=frames)
    s = merge_recording(rec, log=lambda *a: None); m = np.load(rec / "merged_smpl.npz"); bp = m["body_pose"].reshape(T, 21, 3)
    assert np.isfinite(bp[60:120, [0, 1, 3, 4, 6, 7]]).all() and np.isnan(bp[60:120, 5]).all()                 # the legs were seen, the torso was not
    assert not m["export_ok"][60:120].any() and (m["weight"][60:120] == 0).all()                               # dropped whole
    assert m["export_ok"][180:240].all() and np.isnan(bp[180:240, 17]).all() and (m["weight"][180:240, 17] == 0).all() and (m["weight"][180:240, 3] > 0).all()
    assert m["export_ok"][:60].all() and m["export_ok"][120:180].all() and m["export_ok"][240:].all()
    cov = s["coverage_of_lifted_frames"]
    assert abs(cov["legs"] - 1.0) < 1e-9 and abs(cov["torso"] - 0.8) < 1e-9 and abs(cov["legs_and_torso"] - 0.8) < 1e-9 and abs(s["export"]["frac_of_lifted_frames"] - 0.8) < 1e-9
    assert any(f.startswith("info:") and "torso" in f for f in s["flags"]) and s["status"] == "PASS"            # reported, and an info line alone is no warning


def test_a_views_root_glitch_is_repaired_before_merging(tmp_path):
    """One view's root flips for two frames in the middle of a walk and comes back. Left in, the merged heading would
    hop; the merge interpolates across the two frames first and reports it."""
    rng = np.random.default_rng(6); T = 300; frames = np.arange(T)
    body, root, transl, heading = _truth(T, rng); betas = np.zeros(10); conf = np.full(T, 0.9)
    root_l = root.copy(); root_l[150:152] = (R.from_euler("y", 170.0, degrees=True) * R.from_rotvec(root[150:152])).as_rotvec()
    rec = tmp_path / "deliver" / "r"
    _view(rec / "left", "left", frames, _noisy(body, 0.02, rng), root_l, transl, betas, conf, np.ones(T, bool))
    _view(rec / "right", "right", frames, _noisy(body, 0.02, rng), root, transl, betas, conf, np.ones(T, bool))
    s = merge_recording(rec, log=lambda *a: None); m = np.load(rec / "merged_smpl.npz")
    assert s["continuity"]["root_glitch_frames_repaired"] == {"left": 2, "right": 0}
    assert s["continuity"]["root_flips_over_90deg"] == 0 and s["continuity"]["root_turns_over_90deg_in_quarter_s"] == 0
    err = np.abs(((heading_yaw_deg(m["global_orient"]) - heading + 180) % 360) - 180); assert err.max() < 3.0, err.max()
    assert any(f.startswith("info:") and "flipped for an instant" in f for f in s["flags"])


def test_walkway_calibration_recovers_each_views_depth_scale(tmp_path):
    """Two views whose GVHMR positions are wrong by a depth factor about each camera (1.3x and 0.8x), a recording CSV
    reference with the landing foot's position on a mat that is rotated and shifted in the world (2 cm of noise):
    the merge must recover the two scale factors, express the path in the mat frame, and leave a translation closer
    to the truth than the raw one."""
    rng = np.random.default_rng(3); T = 600; frames = np.arange(T); t = frames / 30.0 - 2.0
    body, root, transl, heading = _truth(T, rng); transl = transl - transl.mean(0) + [0.0, 0.95, 0.0]; betas = np.zeros(10)
    cams = {"left": (np.array([6.0, 1.0, 0.5]), 1.3), "right": (np.array([0.0, 2.0, -5.0]), 0.8)}
    rec = tmp_path / "wgvp_output" / "S" / "deliver" / "s_selfpace_mat"; conf = np.full(T, 0.9)
    for name, (c, kerr) in cams.items():
        est = c + kerr * (transl - c)                                      # what a lifter with the wrong body scale reports
        Rc = R.from_euler("y", {"left": 30.0, "right": -70.0}[name], degrees=True)   # an arbitrary camera orientation
        _view(rec / name, name, frames, _noisy(body, 0.02, rng), root, est, betas, conf, np.ones(T, bool))
        z = dict(np.load(rec / name / "smpl_global.npz"))
        np.savez(rec / name / "smpl_incam.npz", transl=Rc.inv().apply(est - c).astype(np.float32), global_orient=z["global_orient"], body_pose=z["body_pose"], betas=z["betas"], frame_idx=frames)
    # the walkway: footfalls every 0.55 s while walking, in a mat frame rotated 25 deg and shifted, with half-step scatter
    from wgvp.merge import feet_positions
    th = np.radians(25.0); o = np.array([1.2, -0.7]); tk = np.arange(0.3, t[-1] - 0.3, 0.55); foot = (np.arange(len(tk)) % 2) + 1          # alternate feet
    Ft = feet_positions(root, body, transl); pk = np.stack([np.interp(tk, t, Ft[:, f - 1, d]) for f, d in zip(foot, [0] * len(tk))], 0)
    pk = np.stack([np.array([np.interp(tk[i], t, Ft[:, foot[i] - 1, 0]) for i in range(len(tk))]), np.array([np.interp(tk[i], t, Ft[:, foot[i] - 1, 2]) for i in range(len(tk))])], 1)
    pk = pk + rng.normal(0, 0.02, pk.shape); d = pk - o; X = np.cos(th) * d[:, 0] - np.sin(th) * d[:, 1]; Y = np.sin(th) * d[:, 0] + np.cos(th) * d[:, 1]   # world -> mat (inverse of the merge's map)
    (tmp_path / "wgvp_output" / "S" / "work" / "imu_ref").mkdir(parents=True)
    np.savez(tmp_path / "wgvp_output" / "S" / "work" / "imu_ref" / "s_selfpace_mat_imu_ref.npz", walkway_t=tk.astype(np.float32), walkway_x_m=X.astype(np.float32), walkway_y_m=Y.astype(np.float32), walkway_foot=foot.astype(np.int8))
    s = merge_recording(rec, log=lambda *a: None); m = np.load(rec / "merged_smpl.npz"); fit = s["walkway"]["fit"]
    assert fit and fit["ok"] and bool(m["transl_calibrated"])
    want = {v: 1.0 / cams[v][1] for v in cams}
    for v, k in zip([str(x) for x in m["views"]], m["walkway_scale"]):
        assert abs(k - want[v]) / want[v] < 0.06, (v, k, want[v])
    def err(tr):                                                       # after one constant offset (the mat / world origin is arbitrary)
        okt = np.isfinite(tr).all(1); d = tr - transl; d = d - np.nanmedian(d[okt], axis=0); return np.linalg.norm(d, axis=1)
    err_raw = err(m["transl"]); err_cal = err(m["transl_scaled"])
    assert np.nanmedian(err_cal) < 0.5 * np.nanmedian(err_raw) and np.nanmedian(err_cal) < 0.06
    assert abs(((fit["mat_yaw_deg"] - 25.0 + 180) % 360) - 180) < 3.0 and np.isfinite(m["transl_mat_video"]).all()
    # the video path in the mat frame matches the fabricated reference up to the scatter
    Xm = np.interp(tk, t, m["transl_mat_video"][:, 0]); assert np.corrcoef(Xm, X)[0, 1] > 0.9 and abs(np.std(Xm) / np.std(X) - 1) < 0.15   # the reference carries half a step of scatter
    # the pelvis path ON THE MAT (footprints + the video pose) covers the walk and beats the video's own path against the truth
    okm = np.isfinite(m["transl_mat"]).all(1); assert okm.mean() > 0.8 and np.isfinite(m["transl_from_walkway"][okm]).all()
    err_mat = err(m["transl_from_walkway"]); assert np.nanmedian(err_mat) < 0.08 and np.nanmedian(err_mat) < 0.5 * np.nanmedian(err_raw), (np.nanmedian(err_mat), np.nanmedian(err_raw))
    # the exported velocity keeps the video path's stride profile; the mat path sets its mean over a stride and a half
    # (frame to frame the mat path carries the footprints' scatter), and nothing steps where the mat path begins or ends
    def hspeed(p): return np.linalg.norm(np.gradient(np.asarray(p, float), t, axis=0)[:, [0, 2]], axis=1)
    sp = np.linalg.norm(m["root_vel_heading_rel_m_s"][:, [0, 2]], axis=1); st = hspeed(transl); sm = hspeed(m["transl_from_walkway"])
    assert np.sqrt(np.nanmean((sp - st) ** 2)) < 0.12 and np.sqrt(np.nanmean((sm - st) ** 2)) > 0.25 and abs(np.nanmean(sp[okm]) - st[okm].mean()) < 0.03
    # (this fabricated walker has no stance phase, its feet travel with the body, so the footfall rows pin the pelvis to ~5 cm;
    # the scale calibration removes the pure depth error exactly here, which no real lift offers)


def test_mirrored_view_is_detected_and_reflected_back(tmp_path):
    """One view is the mirror image of the other (camera software mirroring the picture): its left leg is the
    walker's right leg. The merge must notice from the legs, reflect that view back, and then merge as well as
    with an honest second view."""
    from wgvp.fusion import mirror_body, mirror_root, mirror_transl
    rng = np.random.default_rng(5); T = 400; frames = np.arange(T)
    body, root, transl, heading = _truth(T, rng); betas = np.zeros(10); conf = np.full(T, 0.9)
    Rl = R.from_euler("y", 35.0, degrees=True)
    body_m = mirror_body(_noisy(body, 0.03, rng)); root_m = (Rl * R.from_rotvec(mirror_root(root))).as_rotvec(); transl_m = Rl.apply(mirror_transl(transl)) + [0.5, 0.0, -1.0]
    rec = tmp_path / "deliver" / "s_selfpace"
    _view(rec / "left", "left", frames, _noisy(body, 0.03, rng), root, transl, betas, conf, np.ones(T, bool))
    _view(rec / "right", "right", frames, body_m, root_m, transl_m, betas, conf, np.ones(T, bool))
    s = merge_recording(rec, log=lambda *a: None); m = np.load(rec / "merged_smpl.npz")
    assert s["views_mirrored"] == ["right"] and m["view_mirrored"].tolist() == [False, True] and any("mirror image" in f for f in s["flags"])
    chk = s["mirror_check"]["right"]; assert chk["mirrored"] and chk["mirrored_deg"] < 0.5 * chk["as_is_deg"]
    out = m["body_pose"].reshape(T, 21, 3); legs = np.nanmedian(geodesic(out[:, [0, 1, 3, 4, 6, 7]], body[:, [0, 1, 3, 4, 6, 7]]))
    assert legs < 3.0, legs                                                        # two agreeing views, noise averaged
    # the un-mirrored merge: the same data without the reflection must NOT be flagged
    rec2 = tmp_path / "deliver" / "s2_selfpace"
    _view(rec2 / "left", "left", frames, _noisy(body, 0.03, rng), root, transl, betas, conf, np.ones(T, bool))
    _view(rec2 / "right", "right", frames, _noisy(body, 0.03, rng), (Rl * R.from_rotvec(root)).as_rotvec(), Rl.apply(transl), betas, conf, np.ones(T, bool))
    s2 = merge_recording(rec2, log=lambda *a: None); assert s2["views_mirrored"] == [] and not s2["mirror_check"]["right"]["mirrored"]


def test_agreement_factor_counts_a_dissenting_view_by_its_share(tmp_path):
    """A view that the root rule set aside (its gravity frame wrong, few joints in frame) and whose legs are also
    wrong on a stretch: the merged legs there are the other view's, and the loss weight must say so rather than
    read the dissent as a full disagreement (the along camera of HC901 on a standing walker, legs-only crop)."""
    rng = np.random.default_rng(3); T = 360; frames = np.arange(60, 60 + T)
    body, root, transl, heading = _truth(T, rng); betas = np.zeros(10)
    yaw_l, off_l = -40.0, np.array([1.5, 0.0, -2.0]); Rl = R.from_euler("y", yaw_l, degrees=True)
    body_l = _noisy(body, 0.03, rng); body_l[220:250] = _noisy(body[220:250], 0.5, rng)                      # its legs wrong over 220-250
    root_l = (Rl * R.from_rotvec(root)).as_rotvec()
    root_l[220:250] = (Rl * R.from_euler("x", 40, degrees=True) * R.from_rotvec(root[220:250])).as_rotvec()  # and its gravity frame with them (40 deg: the pitch measured on that camera; past the 30 deg where the root rule starts to act)
    transl_l = Rl.apply(transl) + off_l; conf = np.full(T, 0.9); rec = tmp_path / "deliver" / "s_selfpace"
    _view(rec / "left", "left", frames, body_l, root_l, transl_l, betas, conf, np.ones(T, bool))
    _view(rec / "right", "right", frames, _noisy(body, 0.03, rng), root, transl, betas, conf, np.ones(T, bool))
    k = np.load(rec / "left" / "kp2d_vitpose.npz"); kp = k["kp2d"].copy(); kp[220:250, :11, 1] = -10_000.0        # head to wrists out of its panel there
    np.savez(rec / "left" / "kp2d_vitpose.npz", kp2d=kp, bbox_xyxy=k["bbox_xyxy"], view_box=k["view_box"], frame_idx=frames)
    merge_recording(rec, log=lambda *a: None); m = np.load(rec / "merged_smpl.npz")
    share = m["share"][0][225:245]; w = m["weight"][225:245][:, [3, 4]]; err = geodesic(m["body_pose"].reshape(T, 21, 3), body)
    assert share.mean() < 0.2, share.mean()                                                   # the root rule left it a tenth of the root
    assert np.nanmedian(err[225:245, [3, 4]]) < 3.0, np.nanmedian(err[225:245, [3, 4]])      # the merged knees are the right view's
    assert w.mean() > 0.9, w.mean()                                                            # and the weight says so (the views' knees differ by ~30 deg: read in full, the factor would be ~0.4)
    assert m["weight"][20:100].mean() > 0.9 and np.nanmedian(err[20:100]) < 3.0                # two agreeing views elsewhere: unchanged


def test_stride_swing_comes_from_the_view_that_sees_the_walker_from_the_side(tmp_path):
    """A straight walk seen from behind (left) and from the side (right). The view from behind reads half the knees'
    swing; the side view reads all of it, on a posture 0.1 rad off. The merge keeps the mean posture of the two and
    takes the swing from the side view; without the camera-frame files it can only average the two."""
    T = 300; frames = np.arange(T); t = frames / 30.0; sw = 0.4 * np.sin(2 * np.pi * 0.9 * t); betas = np.zeros(10)

    def body(gain, offset):
        b = np.zeros((T, 21, 3)); b[:, 3, 0] = 0.5 + offset + gain * sw; b[:, 4, 0] = 0.5 + offset - gain * sw; return b
    root = np.zeros((T, 3)); transl = np.c_[np.zeros(T), np.full(T, 0.95), 1.2 * t]; conf = np.full(T, 0.9); vis = np.ones(T, bool)
    rec = tmp_path / "deliver" / "s_selfpace"
    _view(rec / "left", "left", frames, body(0.5, 0.0), root, transl, betas, conf, vis)
    _view(rec / "right", "right", frames, body(1.0, 0.1), root, transl, betas, conf, vis)
    cam = {"left": (np.zeros((T, 3)), transl + [0.0, 0.0, 4.0]),                                                    # behind the walker, looking along the walk
           "right": (np.tile(R.from_euler("y", 90, degrees=True).as_rotvec(), (T, 1)), np.c_[1.2 * t - 6.0, np.zeros(T), np.full(T, 30.0)])}   # far to the side of the walkway
    for v, (r_c, t_c) in cam.items():
        np.savez(rec / v / "smpl_incam.npz", global_orient=r_c.astype(np.float32), transl=t_c.astype(np.float32), frame_idx=frames)

    def knee(s_):
        k = np.load(rec / "merged_smpl.npz")["body_pose"].reshape(T, 21, 3)[60:240, 3, 0]; return float(np.polyfit(sw[60:240], k, 1)[0]), float(k.mean() - 0.5)
    s = merge_recording(rec, log=lambda *a: None); gain, off = knee(s); z = np.load(rec / "merged_smpl.npz")
    assert s["flexion"]["rule"].startswith("two bands") and s["per_view"]["left"]["side_on_mean"] < 0.05 < 0.9 < s["per_view"]["right"]["side_on_mean"]
    assert z["side_on"].shape == (2, T) and gain > 0.93 and abs(off - 0.05) < 0.02, (gain, off)
    for v in cam:
        (rec / v / "smpl_incam.npz").unlink()
    s = merge_recording(rec, log=lambda *a: None); gain, off = knee(s)
    assert s["flexion"]["rule"] == "the views' weighted mean" and 0.7 < gain < 0.8 and abs(off - 0.05) < 0.02, (gain, off)



def test_pelvis_path_takes_each_views_step_across_its_line_of_sight(tmp_path):
    """A straight walk at 1.2 m/s seen from behind (left) and from the side (right). The view from behind reads the
    forward path at half its length (a change of depth); the side view reads it in full. With the camera-frame
    files the merged path takes the forward step from the side view; without them it can only average the two."""
    T = 300; frames = np.arange(T); t = frames / 30.0; betas = np.zeros(10); body = np.zeros((T, 21, 3)); root = np.zeros((T, 3))
    true = np.c_[np.zeros(T), np.full(T, 0.95), 1.2 * t]; half = true * [1.0, 1.0, 0.5]; conf = np.full(T, 0.9); vis = np.ones(T, bool)
    rec = tmp_path / "deliver" / "s_selfpace"
    _view(rec / "left", "left", frames, body, root, half, betas, conf, vis); _view(rec / "right", "right", frames, body, root, true, betas, conf, vis)
    cam = {"left": (np.zeros((T, 3)), true + [0.0, 0.0, 4.0]),
           "right": (np.tile(R.from_euler("y", 90, degrees=True).as_rotvec(), (T, 1)), np.c_[1.2 * t - 6.0, np.zeros(T), np.full(T, 30.0)])}
    for v, (r_c, t_c) in cam.items():
        np.savez(rec / v / "smpl_incam.npz", global_orient=r_c.astype(np.float32), transl=t_c.astype(np.float32), frame_idx=frames)

    def speed():
        z = np.load(rec / "merged_smpl.npz"); return float(np.polyfit(t[30:270], z["transl"][30:270, 2], 1)[0]), float(np.nanmedian(z["root_vel_heading_rel_m_s"][30:270, 2])), z
    s = merge_recording(rec, log=lambda *a: None); v_path, v_out, z = speed()
    assert s["translation"]["rule"].endswith("line of sight") and z["sight_line"].shape == (2, T, 3)
    assert abs(np.nanmedian(np.abs(z["sight_line"][0, :, 2])) - 1.0) < 0.05 and abs(np.nanmedian(np.abs(z["sight_line"][1, :, 0])) - 1.0) < 0.05   # left looks along the walk, right across it
    assert v_path > 1.15 and v_out > 1.15, (v_path, v_out)
    assert not [f for f in s["flags"] if "pelvis position differs" in f and not f.startswith("info:")]
    for v in cam:
        (rec / v / "smpl_incam.npz").unlink()
    s = merge_recording(rec, log=lambda *a: None); v_path, v_out, z = speed()
    assert "no lines of sight" in s["translation"]["rule"] and 0.85 < v_path < 0.95 and np.isnan(z["sight_line"]).all(), v_path


def test_views_position_residual_is_taken_where_the_walker_moves():
    """A walker who stands for most of the recording and whose two views then disagree on the walk: the median over
    all shared frames is set by the standing frames (they agree after any offset); over the moving frames it is not."""
    T = 200; fps = 30.0; z = np.zeros(T); z[120:] = np.arange(80) * (1.2 / fps)                  # stands 4 s, then walks at 1.2 m/s
    t_ref = np.stack([np.zeros(T), np.zeros(T), z], 1); t_v = t_ref * np.array([1.0, 1.0, 1.5])   # the other view reads the walk half again as long
    root = np.zeros((T, 3)); mask = np.ones(T, bool); mov = moving_frames(t_ref, fps)
    assert not mov[:110].any() and mov[130:190].all()
    a = align_to_reference(root, t_ref, root, t_v, mask); b = align_to_reference(root, t_ref, root, t_v, mask, mov)
    assert a["transl_residual_m"] < 0.01 and a["transl_residual_frames"] == "all"
    assert b["transl_residual_m"] > 0.5 and b["transl_residual_frames"] == "moving" and b["offset_m"] == a["offset_m"] and b["yaw_deg"] == a["yaw_deg"]
    still = align_to_reference(root, t_ref, root, t_v, mask, np.zeros(T, bool)); assert still["transl_residual_frames"] == "all"
