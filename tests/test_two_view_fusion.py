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
