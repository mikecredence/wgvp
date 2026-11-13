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
    assert (tmp_path / "out" / "two_view_fusion.md").exists() and json.loads((tmp_path / "out" / "two_view_fusion.json").read_text(encoding="utf-8"))["n_frames"] == T


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
    # the frame right after the gap is not pulled toward the frames before it, nor toward the run's inside: the window shrinks on both sides
    assert np.allclose(sm[150], noisy[150], atol=1e-6) and np.allclose(sm[119], noisy[119], atol=1e-6) and not np.allclose(sm[152], noisy[152], atol=1e-3)
    # a one-frame hole splits a joint's track into two runs: the frames beside it stand alone, their neighbours average three
    holed = noisy.copy(); holed[200, 1] = np.nan; sh = smooth_rotations(holed, 5)
    assert np.isnan(sh[200, 1]).all() and np.isfinite(sh[200, 0]).all() and np.isfinite(sh[[198, 199, 201, 202], 1]).all()
    assert np.allclose(sh[199, 1], holed[199, 1], atol=1e-6) and np.allclose(sh[201, 1], holed[201, 1], atol=1e-6)
    assert np.allclose(sh[:, 0], sm[:, 0], atol=1e-9, equal_nan=True)                                       # the other joint is untouched


def test_a_path_keeps_its_last_steps_under_the_centred_moving_average():
    """A walker crossing a run's end at constant speed: the one-sided window halves the last two steps of the path
    (a smoother for slow trends), the centred one keeps every step."""
    from wgvp.fusion import moving_average_in_runs
    x = np.arange(40, dtype=float)[:, None] * [0.04, 0.0, 0.01]; x[18:22] = np.nan
    one, cen = moving_average_in_runs(x, 5), moving_average_in_runs(x, 5, centred=True)
    assert (np.isfinite(one) == np.isfinite(x)).all() and (np.isfinite(cen) == np.isfinite(x)).all()
    assert np.allclose(cen[np.isfinite(x[:, 0])], x[np.isfinite(x[:, 0])])                                 # a straight path is its own centred average, ends included
    assert np.isclose(one[1, 0] - one[0, 0], 0.02) and np.isclose(one[17, 0] - one[16, 0], 0.02)          # the one-sided ends move at half speed
    rng = np.random.default_rng(0); n = x + rng.normal(0, 0.01, x.shape); k = np.r_[3:15, 25:37]
    assert np.allclose(moving_average_in_runs(n, 5, centred=True)[k], moving_average_in_runs(n, 5)[k])     # away from the ends the two are the same average
    assert np.allclose(moving_average_in_runs(n[:, 0], 5, centred=True), moving_average_in_runs(n, 5, centred=True)[:, 0], equal_nan=True)


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


def test_export_mask_requires_every_leg_and_torso_joint_in_runs():
    """The export rule: a frame with a leg joint or a torso joint (spine, neck, head) not output is dropped whole;
    a frame with only an arm joint missing is kept."""
    from wgvp.fusion import LEGS, TORSO_LEGS, export_mask
    assert TORSO_LEGS == [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14]
    T = 130; m = np.zeros((T, 21, 3)); m[:10] = np.nan; m[40:48, 3] = np.nan; m[70:80, 12] = np.nan; m[70:80, 19] = np.nan   # nothing / left knee / a collar and a wrist (arms: not required)
    m[100:110, 5] = np.nan                                                                                     # a spine joint: the torso is required
    ok = export_mask(m, min_run=15)
    assert not ok[:10].any() and ok[10:40].all() and not ok[40:48].any() and ok[48:100].all() and not ok[100:110].any() and ok[110:].all()
    legs_only = export_mask(m, LEGS, min_run=15)                                                               # the older, laxer rule is still available by name
    assert legs_only[100:110].all() and not legs_only[40:48].any()
    m2 = np.zeros((T, 21, 3)); m2[:50] = np.nan; m2[60:] = np.nan                                              # a 10-frame run
    assert not export_mask(m2, min_run=15).any() and export_mask(m2, min_run=10).sum() == 10


def test_repair_root_glitches_interpolates_a_flip_that_comes_back_and_leaves_turns_alone():
    """A lift's root that flips for two frames and comes back is interpolated across; a turn at a human rate and a
    step that does not come back (a snap: fuse_root's business) are left as they are."""
    from scipy.spatial.transform import Rotation as R
    from wgvp.fusion import heading_yaw_deg, repair_root_glitches
    T = 240; t = np.arange(T) / 30.0; h = 10.0 + 180.0 * np.clip((t - 2.0) / 1.2, 0.0, 1.0)                    # a 180-degree turn in 1.2 s (150 deg/s)
    def roots(hh): return np.stack([(R.from_euler("y", x, degrees=True) * R.from_euler("x", 6.0, degrees=True)).as_rotvec() for x in hh])
    clean = roots(h); out, fixed = repair_root_glitches(clean, 30.0)
    assert not fixed.any() and np.allclose(out, clean)                                                          # the turn itself is no glitch
    hg = h.copy(); hg[75:77] += 170.0                                                                         # two frames on the far side, mid-turn
    out, fixed = repair_root_glitches(roots(hg), 30.0)
    assert fixed[75:77].all() and fixed.sum() == 2
    err = np.abs(((heading_yaw_deg(out) - h + 180) % 360) - 180); assert err.max() < 3.0, err.max()
    hs = h.copy(); hs[150:] += 120.0                                                                          # a step that stays: not repaired
    out, fixed = repair_root_glitches(roots(hs), 30.0)
    assert not fixed.any() and np.allclose(out, roots(hs))
    gap = roots(hg); gap[10:20] = np.nan                                                                      # gaps stay gaps
    out, fixed = repair_root_glitches(gap, 30.0)
    assert np.isnan(out[10:20]).all() and fixed.sum() == 2


def test_fuse_root_turn_has_no_jitter_when_two_views_sit_near_the_disagreement_threshold():
    """A turn that one view's lift follows a third of a second late: the two roots sit 30-45 deg apart for the
    length of the turn, right where the arbitration switches on. The fused heading must turn smoothly (no hop from
    one frame to the next beyond the turn's own rate), and the smoothed arbitration must not be rougher than the
    causal one it replaces."""
    from scipy.spatial.transform import Rotation as R
    from wgvp.fusion import fuse_root, heading_yaw_deg
    T = 300; t = np.arange(T) / 30.0; rng = np.random.default_rng(3)
    h_a = 20.0 + 180.0 * np.clip((t - 5.0) / 1.5, 0.0, 1.0); h_b = 20.0 + 180.0 * np.clip((t - 5.3) / 1.5, 0.0, 1.0)   # 120 deg/s; b lags by 0.3 s = 36 deg
    def roots(h): return np.stack([(R.from_euler("y", hh + rng.normal(0, 1.5), degrees=True) * R.from_euler("x", 5.0, degrees=True)).as_rotvec() for hh in h])
    a, b = roots(h_a), roots(h_b); usable = [np.ones((T, 21), bool), np.ones((T, 21), bool)]; lifted = [np.ones(T, bool), np.ones(T, bool)]
    def jerk(arb_smooth):
        fused, _ = fuse_root([a, b], usable, lifted, ramp=15, arb_smooth=arb_smooth)
        hf = np.degrees(np.unwrap(np.radians(heading_yaw_deg(fused)))); return hf, np.abs(np.diff(hf, 2))[int(4.5 * 30):int(7.5 * 30)]
    hf, j1 = jerk(True); _, j0 = jerk(False)
    assert np.abs(np.diff(hf)).max() < 12.0, np.abs(np.diff(hf)).max()                                          # 120 deg/s is 4 deg per frame; nothing like a hand-over hop
    assert np.percentile(j1, 95) <= np.percentile(j0, 95) + 1e-9, (np.percentile(j1, 95), np.percentile(j0, 95))
    assert hf[-1] - hf[0] > 170.0 and (np.minimum(h_a, h_b) - 8 <= hf).all() and (hf <= np.maximum(h_a, h_b) + 8).all()   # between the two views throughout


def test_side_consistent_undoes_a_left_right_flip_for_display():
    """The demo's overlay helper: when a 2D estimator swaps the left and right leg labels mid-sequence, the drawn
    skeleton keeps the assignment that continues the previous frame; a swap that is a genuine crossing (the two
    legs at the same place) is left alone, and nothing but the arm and leg chains is touched."""
    from wgvp.demo import side_consistent
    T = 20; kp = np.full((T, 17, 3), np.nan); kp[:, :, 2] = 0.9
    for t_ in range(T):                                                   # left leg ahead at x=100, right leg behind at x=60, both moving slowly
        kp[t_, [11, 13, 15], :2] = [[100 + t_, 50], [100 + t_, 100], [100 + t_, 150]]; kp[t_, [12, 14, 16], :2] = [[60 + t_, 50], [60 + t_, 100], [60 + t_, 150]]
        kp[t_, 0, :2] = [80, 10]
    flipped = kp.copy(); flipped[8:, [11, 13, 15]] = kp[8:, [12, 14, 16]]; flipped[8:, [12, 14, 16]] = kp[8:, [11, 13, 15]]   # the estimator swaps sides from frame 8 on
    fixed = side_consistent(flipped)
    assert np.allclose(fixed[:, [11, 13, 15], :2], kp[:, [11, 13, 15], :2]) and np.allclose(fixed[:, [12, 14, 16], :2], kp[:, [12, 14, 16], :2])
    assert np.allclose(fixed[:, 0], kp[:, 0]) and np.allclose(fixed[:, [5, 7, 9]], flipped[:, [5, 7, 9]], equal_nan=True)
    assert np.allclose(side_consistent(kp), kp, equal_nan=True)              # a consistent sequence is unchanged


def test_fuse_root_follows_continuity_when_one_view_turns_the_wrong_way():
    """A 180-degree turn: one view's lift turns with the truth at a human rate; the other snaps through the
    front/back ambiguity to the far side in a few frames and then sits 120 deg from the truth until the turn is
    over. With equal weights their mean would be meaningless; the fusion must discount the snap and stay with the
    view it was following, so the fused heading tracks the truth with no jump."""
    from scipy.spatial.transform import Rotation as R
    from wgvp.fusion import fuse_root, heading_yaw_deg
    T = 300; t = np.arange(T) / 30.0; turn = np.clip((t - 5.0) / 1.5, 0.0, 1.0)                    # the turn runs from 5.0 to 6.5 s
    h_truth = 20.0 + 180.0 * turn; h_wrong = 20.0 - 180.0 * turn                                   # the mirror path: the same end point, the other way round
    snap = np.clip((t - 5.4) / 0.13, 0.0, 1.0)                                                       # at 5.4 s the lift snaps from the true path onto the mirror path in 4 frames
    h_snap = h_truth + (h_wrong - h_truth) * snap
    rng = np.random.default_rng(0)
    def roots(h): return np.stack([(R.from_euler("y", hh, degrees=True) * R.from_euler("x", 5.0 + rng.normal(0, 1.0), degrees=True)).as_rotvec() for hh in h])
    a, b = roots(h_truth), roots(h_snap); usable = [np.ones((T, 21), bool), np.ones((T, 21), bool)]; lifted = [np.ones(T, bool), np.ones(T, bool)]
    fused, share = fuse_root([a, b], usable, lifted, ramp=15)
    hf = heading_yaw_deg(fused); err = np.abs(((hf - h_truth + 180) % 360) - 180); step = np.abs(((np.diff(hf) + 180) % 360) - 180)
    assert err.max() < 25.0, err.max()
    assert step.max() < 30.0, step.max()
    assert share[0][(t > 5.55) & (t < 6.2)].mean() > 0.9                                              # the view that continues the motion carries the turn
    assert abs(share[0][t > 7.5].mean() - 0.5) < 0.1                                                  # once the views agree again, both count


def test_fuse_root_drops_the_view_whose_coverage_ends_inside_a_wrong_way_turn():
    """The wrong-way turn that never snaps: the heavier view turns the wrong way round and then simply leaves
    (the walker walks out of its frame mid-turn). Nothing resolves the disagreement, and at the hand-off the fused
    heading would jump by the whole gap onto the surviving view. The rule treats a view whose coverage ends inside
    an episode like one that snaps, so the surviving view carries the turn from the start of the episode."""
    from scipy.spatial.transform import Rotation as R
    from wgvp.fusion import fuse_root, heading_yaw_deg
    T = 300; t = np.arange(T) / 30.0; turn = np.clip((t - 5.0) / 1.5, 0.0, 1.0)                    # the turn runs from 5.0 to 6.5 s
    h_truth = 20.0 + 180.0 * turn; h_wrong = 20.0 - 90.0 * turn                                    # the wrong view turns the other way, slower
    rng = np.random.default_rng(2)
    def roots(h): return np.stack([(R.from_euler("y", hh, degrees=True) * R.from_euler("x", 5.0 + rng.normal(0, 1.0), degrees=True)).as_rotvec() for hh in h])
    a, b = roots(h_truth), roots(h_wrong)
    us_a = np.zeros((T, 21), bool); us_a[:, :14] = True; us_b = np.ones((T, 21), bool)               # the wrong view sees the whole body, the right one 14 joints
    lifted_b = t < 6.0; b[~lifted_b] = np.nan; us_b[~lifted_b] = False                                # and its lift ends mid-turn, 150 degrees off
    fused, share = fuse_root([a, b], [us_a, us_b], [np.ones(T, bool), lifted_b], ramp=15)
    hf = heading_yaw_deg(fused); err = np.abs(((hf - h_truth + 180) % 360) - 180); step = np.abs(((np.diff(hf) + 180) % 360) - 180)
    assert err.max() < 25.0, err.max()
    assert step.max() < 30.0, step.max()
    assert share[0][(t > 5.4) & (t < 6.0)].min() > 0.99                                             # the surviving view carries the turn before the hand-off too
    assert np.allclose(share[0][t >= 6.0], 1.0)
    c = roots(h_truth); c[~lifted_b] = np.nan                                                         # a view that leaves while agreeing: the rule is silent
    fused2, share2 = fuse_root([a, c], [us_a, us_b], [np.ones(T, bool), lifted_b], ramp=15)
    assert np.abs(((heading_yaw_deg(fused2) - h_truth + 180) % 360) - 180).max() < 3.0
    assert share2[1][(t > 1.0) & (t < 5.0)].mean() > 0.6                                             # and the heavier view keeps its weight while it is there


def test_fuse_root_drops_the_view_that_snaps_at_the_end_of_a_wrong_way_turn():
    """The other failure of a 180-degree turn: one view turns with the truth at a human rate, the other turns the
    wrong way round at a human rate too, so nothing distinguishes them while they turn, and only gives itself
    away by snapping onto the true heading when the turn is over. The causal terms cannot help (the merge had
    locked onto the heavier, wrong view by then); the offline snap rule must drop the snapping view for the
    whole episode, so the fused heading follows the truth with no jump."""
    from scipy.spatial.transform import Rotation as R
    from wgvp.fusion import fuse_root, heading_yaw_deg
    T = 300; t = np.arange(T) / 30.0; turn = np.clip((t - 5.0) / 1.5, 0.0, 1.0)                    # the turn runs from 5.0 to 6.5 s
    h_truth = 20.0 + 180.0 * turn; h_wrong = 20.0 - 90.0 * turn                                    # the wrong view turns the other way, slower
    snap = np.clip((t - 6.3) / 0.1, 0.0, 1.0)                                                       # and snaps onto the truth in 3 frames at 6.3 s
    h_snap = h_wrong + (h_truth - h_wrong) * snap
    rng = np.random.default_rng(1)
    def roots(h): return np.stack([(R.from_euler("y", hh, degrees=True) * R.from_euler("x", 5.0 + rng.normal(0, 1.0), degrees=True)).as_rotvec() for hh in h])
    a, b = roots(h_truth), roots(h_snap)
    us_a = np.zeros((T, 21), bool); us_a[:, :14] = True; us_b = np.ones((T, 21), bool)               # the wrong view sees the whole body, the right one 14 joints
    fused, share = fuse_root([a, b], [us_a, us_b], [np.ones(T, bool), np.ones(T, bool)], ramp=15)
    hf = heading_yaw_deg(fused); err = np.abs(((hf - h_truth + 180) % 360) - 180); step = np.abs(((np.diff(hf) + 180) % 360) - 180)
    assert err.max() < 25.0, err.max()
    assert step.max() < 30.0, step.max()
    assert share[0][(t > 5.6) & (t < 6.3)].min() > 0.99                                             # the view that did not snap carries the whole turn
    assert abs(share[0][t > 7.5].mean() - (0.44 / (1 + 0.44))) < 0.1                                # afterwards the usable fractions decide again
    fused2, _ = fuse_root([a, a], [us_a, us_b], [np.ones(T, bool), np.ones(T, bool)], ramp=15)      # no snap anywhere: the rule is silent
    assert np.abs(((heading_yaw_deg(fused2) - h_truth + 180) % 360) - 180).max() < 1e-6
