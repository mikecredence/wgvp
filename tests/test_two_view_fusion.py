"""The two-view fusion analysis on a fabricated truth: two noisy views of one pose sequence, one of them badly wrong
over a stretch where its (fake) keypoint confidence is low; the confidence-weighted fusion must beat both single
views there, and the oracle pick must beat everything."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.two_view_fusion import main as fusion_main   # noqa: E402


def _view(d: Path, frames, body, root, conf, visible):
    d.mkdir(parents=True)
    T = len(frames)
    np.savez(d / "smpl_global.npz", frame_idx=frames, imu_t_s=frames / 30.0, body_pose=body.reshape(T, 63).astype(np.float32), global_orient=root.astype(np.float32),
             betas=np.zeros((T, 10), np.float32), transl=np.zeros((T, 3), np.float32), visible=visible, edge_truncated=np.zeros(T, bool), chunk_id=np.zeros(T, np.int16), pose_step_deg=np.zeros(T, np.float32))
    kp = np.zeros((T, 17, 3), np.float32); kp[:, :, 2] = conf[:, None]
    np.savez(d / "kp2d_vitpose.npz", kp2d=kp, bbox_xyxy=np.zeros((T, 4), np.float32), frame_idx=frames)


def test_fusion_beats_single_views_where_one_is_damaged(tmp_path):
    rng = np.random.default_rng(0); T = 240; frames = np.arange(90, 90 + T)
    gt_body = np.cumsum(rng.normal(0, 0.01, (T, 21, 3)), axis=0)                      # a smooth pose sequence
    gt_root = np.tile(R.from_euler("y", 20, degrees=True).as_rotvec(), (T, 1)).astype(np.float32)
    joints_in = np.ones((T, 24), bool); j_left = joints_in.copy(); j_left[100:160, [15, 16, 17]] = False   # the along view loses head+shoulders for 2 s
    gt = tmp_path / "gt.npz"
    np.savez(gt, frame_idx=frames, imu_t_s=frames / 30.0, pose_aa=np.concatenate([gt_root, gt_body.reshape(T, 63), np.zeros((T, 6))], 1).astype(np.float32),
             transl=np.zeros((T, 3), np.float32), joints3d=np.zeros((T, 24, 3), np.float32), left_joints_in=j_left, right_joints_in=joints_in)
    noise = lambda sd, n: R.from_rotvec(rng.normal(0, sd, (n * 21, 3))).as_rotvec().reshape(n, 21, 3)
    compose = lambda a, b: (R.from_rotvec(b.reshape(-1, 3)) * R.from_rotvec(a.reshape(-1, 3))).as_rotvec().reshape(a.shape)
    left = compose(gt_body, noise(0.05, T)); left[100:160] = compose(gt_body[100:160], noise(0.6, 60))      # damaged stretch
    right = compose(gt_body, noise(0.05, T))
    conf_l = np.full(T, 0.9); conf_l[100:160] = 0.1; conf_r = np.full(T, 0.9)
    # each view's GVHMR world frame has its own heading
    root_l = (R.from_euler("y", -35, degrees=True) * R.from_rotvec(gt_root)).as_rotvec(); root_r = (R.from_euler("y", 50, degrees=True) * R.from_rotvec(gt_root)).as_rotvec()
    deliver = tmp_path / "wgvp_output" / "HC901" / "deliver" / "hc901_selfpace"
    _view(deliver / "left", frames, left, root_l, conf_l, np.ones(T, bool)); _view(deliver / "right", frames, right, root_r, conf_r, np.ones(T, bool))
    res = fusion_main([str(tmp_path), "--gt", str(gt), "--out", str(tmp_path / "out"), "--quiet"])
    o = res["overall"]
    assert o["right"]["all"] < 6 and o["left"]["all"] < 8                       # the undamaged parts dominate the medians
    dmg = res["fused_by_along_visibility"]["shoulders cut"]
    left_dmg = res["by_visibility"]["left"]["shoulders cut"]["all"]
    assert left_dmg > 25                                                        # the damage is real
    assert dmg["confidence-weighted"]["all"] < 0.5 * dmg["mean"]["all"] < left_dmg   # weighting by confidence rescues it; a plain mean only halves it
    assert dmg["oracle pick"]["all"] <= dmg["confidence-weighted"]["all"] + 0.5
    assert abs(res["yaw_offset_deg"]["left"] - 35) <= 1.5 and abs(res["yaw_offset_deg"]["right"] + 50) <= 1.5   # heading offsets recovered
    assert (tmp_path / "out" / "two_view_fusion.md").exists() and json.loads((tmp_path / "out" / "two_view_fusion.json").read_text())["n_frames"] == T


def test_smooth_rotations_reduces_jitter_and_respects_gaps():
    """The merge smoother: a short moving average in rotation space that damps frame-to-frame jitter, keeps the
    underlying motion, and treats NaN gaps as hard boundaries (no averaging across them, no filling)."""
    from tools.two_view_fusion import smooth_rotations, step_stats
    rng = np.random.default_rng(0); T = 300; t = np.arange(T) / 30.0
    clean = np.zeros((T, 2, 3)); clean[:, 0, 0] = 0.6 * np.sin(2 * np.pi * 0.6 * t)          # a stride-rate swing
    noisy = clean + rng.normal(0, 0.08, clean.shape)                                           # ~4.6 deg lifter jitter per axis (GVHMR's p99 step is ~15 deg)
    noisy[120:150] = np.nan                                                                     # a gap in the output
    noisy[150, 0, 0] += 1.0                                                                     # a genuine different pose after the gap
    sm = smooth_rotations(noisy, 5)
    assert np.isnan(sm[120:150]).all() and np.isfinite(sm[:120]).all() and np.isfinite(sm[150:]).all()   # gaps stay gaps, nothing else lost
    resid = lambda x: np.degrees(np.nanstd(x[:120, 0, 0] - clean[:120, 0, 0]))                            # jitter around the true motion
    assert resid(sm) < 0.6 * resid(noisy)                                                                   # damped by at least 40 %
    assert step_stats(sm)["p99_deg"] < step_stats(noisy)["p99_deg"]                                        # and steadier frame to frame
    err = lambda x: np.degrees(np.nanmean(np.abs(x[:120, 0, 0] - clean[:120, 0, 0])))
    assert err(sm) < err(noisy)                                                                             # closer to the clean motion, not just smoother
    # the frame right after the gap must not be pulled toward the frames before it
    assert abs(sm[150, 0, 0] - noisy[150:155, 0, 0].mean()) < abs(sm[150, 0, 0] - noisy[115:120, 0, 0].mean())


def test_fuse_root_is_continuous_and_prefers_the_view_that_sees_more():
    """The merge's root: a view that sees few joints and disagrees with the one that sees the whole body barely
    moves the root, cannot step it when it enters or leaves, and carries it alone when it is the only view."""
    from tools.two_view_fusion import fuse_root, step_stats
    T = 200; good = np.tile(R.from_euler("x", 5, degrees=True).as_rotvec(), (T, 1)); bad = np.tile(R.from_euler("x", 50, degrees=True).as_rotvec(), (T, 1))
    us_good = np.ones((T, 21), bool); us_bad = np.zeros((T, 21), bool); us_bad[:, :5] = True                 # sees 5 of 21 joints
    lifted_good = np.ones(T, bool); lifted_bad = np.zeros(T, bool); lifted_bad[60:140] = True                # enters and leaves mid-run
    out, share = fuse_root([good, bad], [us_good, us_bad], [lifted_good, lifted_bad])
    assert np.isfinite(out).all()
    assert np.degrees((R.from_rotvec(out).inv() * R.from_rotvec(good)).magnitude()).max() < 2.0
    assert step_stats(out[:, None, :])["p99_deg"] < 0.5
    assert share[1, 60:140].max() < 0.05 and share[0, :60].min() == 1.0
    out2, share2 = fuse_root([good, good], [us_good, us_good], [lifted_good, lifted_good])                  # agreeing equals: an even average
    assert np.allclose(share2, 0.5) and np.degrees((R.from_rotvec(out2).inv() * R.from_rotvec(good)).magnitude()).max() < 1e-6
    out3, share3 = fuse_root([good, bad], [us_good, us_bad], [np.zeros(T, bool), lifted_bad])                # alone: it carries the root
    assert np.isnan(out3[:60]).all() and np.allclose(out3[60:140], bad[60:140], atol=1e-6) and np.allclose(share3[1, 60:140], 1.0)


def test_export_mask_requires_every_leg_joint_in_runs():
    from tools.two_view_fusion import LEGS, export_mask
    T = 100; m = np.zeros((T, 21, 3)); m[:10] = np.nan; m[40:48, 3] = np.nan; m[70:80, 12] = np.nan           # nothing / left knee missing / a collar missing (not required)
    ok = export_mask(m, LEGS, min_run=15)
    assert not ok[:10].any() and ok[10:40].all() and not ok[40:48].any() and ok[48:].all()
    m2 = np.zeros((T, 21, 3)); m2[:50] = np.nan; m2[60:] = np.nan                                              # a 10-frame run
    assert not export_mask(m2, LEGS, 15).any() and export_mask(m2, LEGS, 10).sum() == 10


def test_side_consistent_undoes_a_left_right_flip_for_display():
    """The demo's overlay helper: when a 2D estimator swaps the left and right leg labels mid-sequence, the drawn
    skeleton keeps the assignment that continues the previous frame; a swap that is a genuine crossing (the two
    legs at the same place) is left alone, and nothing but the arm and leg chains is touched."""
    from tools.make_demo import side_consistent
    T = 20; kp = np.full((T, 17, 3), np.nan); kp[:, :, 2] = 0.9
    for t_ in range(T):                                                   # left leg ahead at x=100, right leg behind at x=60, both moving slowly
        kp[t_, [11, 13, 15], :2] = [[100 + t_, 50], [100 + t_, 100], [100 + t_, 150]]; kp[t_, [12, 14, 16], :2] = [[60 + t_, 50], [60 + t_, 100], [60 + t_, 150]]
        kp[t_, 0, :2] = [80, 10]
    flipped = kp.copy(); flipped[8:, [11, 13, 15]] = kp[8:, [12, 14, 16]]; flipped[8:, [12, 14, 16]] = kp[8:, [11, 13, 15]]   # the estimator swaps sides from frame 8 on
    fixed = side_consistent(flipped)
    assert np.allclose(fixed[:, [11, 13, 15], :2], kp[:, [11, 13, 15], :2]) and np.allclose(fixed[:, [12, 14, 16], :2], kp[:, [12, 14, 16], :2])
    assert np.allclose(fixed[:, 0], kp[:, 0]) and np.allclose(fixed[:, [5, 7, 9]], flipped[:, [5, 7, 9]], equal_nan=True)
    assert np.allclose(side_consistent(kp), kp, equal_nan=True)              # a consistent sequence is unchanged
