"""Small helpers of the fusion library: the mirror map (exact for the SMPL-X template by forward kinematics) and the
collapsed-legs test of a 2D skeleton."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.fusion import (L_FOOT, R_FOOT, collapsed_legs, fk_joints, flexion_two_band, fuse_path, mirror_body, mirror_check, mirror_root,   # noqa: E402
                         mirror_transl, side_on_score, sight_line_world)


def test_mirror_map_reflects_the_body():
    rng = np.random.default_rng(0); T = 200
    body = rng.normal(0, 0.4, (T, 21, 3)); root = R.from_euler("yx", np.c_[rng.uniform(-180, 180, T), rng.normal(0, 5, T)], degrees=True).as_rotvec(); tr = rng.normal(0, 1, (T, 3))
    J = fk_joints(root, body, tr); Jm = fk_joints(mirror_root(root), mirror_body(body), mirror_transl(tr))
    assert np.abs(Jm[:, L_FOOT] - J[:, R_FOOT] * [-1, 1, 1]).max() < 0.05 and np.abs(Jm[:, R_FOOT] - J[:, L_FOOT] * [-1, 1, 1]).max() < 0.05   # template asymmetry: mm
    assert np.abs(mirror_body(mirror_body(body)) - body).max() == 0.0
    chk = mirror_check(body, mirror_body(body), np.ones(T, bool)); assert chk["mirrored"] and chk["mirrored_deg"] < 1e-6 < chk["as_is_deg"]
    assert not mirror_check(body, body, np.ones(T, bool))["mirrored"] and not mirror_check(body, mirror_body(body), np.arange(T) < 20)["mirrored"]   # too few frames: no verdict


def test_collapsed_legs_flags_both_legs_on_one_leg():
    kp = np.zeros((4, 17, 3)); kp[:, :, 2] = 0.9
    kp[:, 13] = [100, 200, 0.9]; kp[:, 15] = [100, 300, 0.9]                       # left knee / ankle
    kp[0, 14] = [160, 200, 0.9]; kp[0, 16] = [170, 300, 0.9]                        # frame 0: right leg apart
    kp[1, 14] = [101, 201, 0.9]; kp[1, 16] = [100, 301, 0.9]                        # frame 1: both legs on one leg
    kp[2, 14] = [160, 200, 0.9]; kp[2, 16] = [101, 300, 0.9]                        # frame 2: ankles together, knees apart (a crossing)
    kp[3, 14] = [101, 201, 0.1]; kp[3, 16] = [100, 301, 0.9]                        # frame 3: collapsed but not confident
    assert collapsed_legs(kp).tolist() == [False, True, False, False]


def test_usable_joints_shoulders_get_a_thinner_margin_at_the_top_edge_only():
    """Shoulder keypoints 6 px below the panel top keep the trunk usable (only the head lies beyond them); hip
    keypoints 6 px below the top (a body cut at the waist) do not keep the legs, and a shoulder 6 px from a side
    edge does not keep its arm."""
    from wgvp.fusion import ARMS, LEGS, SPINE, usable_joints_rt
    box = (100.0, 0.0, 620.0, 400.0); vis = np.ones(1, bool)
    kp = np.zeros((1, 17, 3)); kp[:, :, 0] = 360.0; kp[:, :, 1] = 200.0; kp[:, :, 2] = 0.9
    assert usable_joints_rt(kp, box, vis).all()
    top = kp.copy(); top[:, [5, 6], 1] = 6.0
    u = usable_joints_rt(top, box, vis); assert u[0, SPINE].all() and u[0, LEGS].all() and u[0, ARMS].all()
    assert not usable_joints_rt(top, box, vis, shoulder_top_margin=8.0)[0, SPINE].any()
    edge = kp.copy(); edge[:, [5, 6], 1] = 3.0; assert not usable_joints_rt(edge, box, vis)[0, SPINE].any()
    hips = kp.copy(); hips[:, [11, 12], 1] = 6.0
    u = usable_joints_rt(hips, box, vis); assert not u[0, [0, 1]].any() and not u[0, [2, 5]].any() and u[0, 8]
    side = kp.copy(); side[:, 5, 0] = 106.0
    u = usable_joints_rt(side, box, vis); assert not u[0, SPINE].any() and not u[0, [12, 15]].any() and u[0, [13, 16]].all()


def test_usable_joints_shoulder_margin_has_hysteresis():
    """A shoulder riding the 4-px top margin (3.7-4.3 px) keeps the trunk through the whole stretch once it has been
    4 px inside; a stretch that never reaches 4 px, a frame under the 2-px hold, an unseen frame and a jump in the
    frame index all end it."""
    from wgvp.fusion import SPINE, hold_runs, usable_joints_rt
    box = (100.0, 0.0, 620.0, 400.0)
    def trunk(y, vis=None, **kw):
        kp = np.zeros((len(y), 17, 3)); kp[:, :, 0] = 360.0; kp[:, :, 1] = 200.0; kp[:, :, 2] = 0.9; kp[:, [5, 6], 1] = np.asarray(y, float)[:, None]
        return usable_joints_rt(kp, box, np.ones(len(y), bool) if vis is None else np.asarray(vis, bool), **kw)[:, SPINE].all(1)
    ride = [3.8, 4.2, 3.9, 3.7, 4.3, 3.6]
    assert trunk(ride).all() and trunk(ride, shoulder_top_hold=4.0).tolist() == [False, True, False, False, True, False]
    assert not trunk([3.8, 3.9, 3.7]).any()                                             # never 4 px inside
    assert trunk([4.2, 3.0, 1.5, 3.0, 3.9]).tolist() == [True, True, False, False, False]  # under the hold: the run ends
    assert trunk([4.2, 3.0, 3.0, 3.0], vis=[1, 1, 0, 1]).tolist() == [True, True, False, False]
    assert trunk([4.2, 3.0, 3.0, 3.0], frame_idx=[10, 11, 40, 41]).tolist() == [True, True, False, False]
    assert hold_runs([1, 1, 0, 1, 1, 1], [0, 1, 0, 0, 0, 0]).tolist() == [True, True, False, False, False, False]
    assert hold_runs([1, 1, 1, 1], [1, 0, 0, 0], breaks=[0, 0, 1, 0]).tolist() == [True, True, False, False]


def test_rest_held_frames_count_as_seen_only_when_the_keypoints_confirm_the_subject():
    """A frame lifted on a box held at rest is used when ViTPose finds the body where the neighbouring detection has
    it; a held box with faint keypoints (an occluder), or with the body somewhere else, is not."""
    from wgvp.fusion import rest_held_seen, seen_frames
    T = 40; kp = np.zeros((T, 17, 3)); kp[:, :, 0] = 300.0; kp[:, :, 1] = np.linspace(100, 400, 17); kp[:, :, 2] = 0.9
    bbox = np.tile([250.0, 80.0, 350.0, 420.0], (T, 1)); vis = np.ones(T, bool); held = np.zeros(T, bool)
    vis[:10] = False; held[:10] = True                       # held before the first detection: same keypoints
    vis[15:20] = False; held[15:20] = True; kp[15:20, :, 2] = 0.2          # held, keypoints faint
    vis[25:30] = False; held[25:30] = True; kp[25:30, :, 0] += 40.0        # held, body 40 px (0.12 box heights) away
    vis[35:38] = False                                                      # a bridged gap that is not held
    ok = rest_held_seen(vis, held, kp, bbox)
    assert ok[:10].all() and not ok[10:].any()
    z = {"visible": vis, "rest_held": held}; k = {"kp2d": kp, "bbox_xyxy": bbox}
    seen = seen_frames(z, k); assert seen[:15].all() and not seen[15:20].any() and not seen[25:30].any() and not seen[35:38].any() and seen[38:].all()
    assert (seen_frames({"visible": vis}, k) == vis).all()                  # a lift from before the field
    kp[25:30, :, 0] -= 30.0; assert rest_held_seen(vis, held, kp, bbox)[25:30].all()   # 10 px = 0.03 box heights: the same place


def test_side_on_score_is_one_from_the_side_and_zero_head_on():
    eye = np.zeros((1, 3)); quarter = R.from_euler("y", 90, degrees=True).as_rotvec()[None]; ahead = np.array([[0.0, 0.1, 5.0]])
    assert side_on_score(eye, ahead)[0] < 1e-3                                   # walking straight away from the camera
    assert side_on_score(quarter, ahead)[0] > 0.999                              # the camera looks along the hips' axis
    assert side_on_score(eye, np.array([[5.0, 0.0, 0.2]]))[0] > 0.99             # the ray counts, not the optical axis: a walker far to one side is seen from the side
    rng = np.random.default_rng(3); aa = rng.normal(0, 1.0, (50, 3)); p = rng.normal(0, 1, (50, 3)) + [0, 0, 6.0]; s = side_on_score(aa, p)
    M = np.diag([-1.0, 1.0, 1.0]); aa_m = R.from_matrix(M @ R.from_rotvec(aa).as_matrix() @ M).as_rotvec()        # the same scene in a mirrored picture
    assert ((s >= 0) & (s <= 1)).all() and np.abs(side_on_score(aa_m, p * [-1, 1, 1]) - s).max() < 1e-9
    aa[7] = np.nan; assert np.isnan(side_on_score(aa, p)[7]) and np.isfinite(side_on_score(aa, p)[8])


def test_flexion_two_band_keeps_the_posture_of_one_estimate_and_the_swing_of_the_other():
    T = 300; t = np.arange(T) / 30.0; swing = np.sin(2 * np.pi * 1.0 * t); abd = R.from_euler("z", 0.2)

    def pose(flex, rest):
        return (rest * R.from_euler("x", flex)).as_rotvec()[:, None, :].repeat(2, 1)
    base = pose(0.3 + 0.25 * swing, abd)                 # the right posture and the right abduction, half the stride's swing
    alt = pose(0.5 + 0.5 * swing, R.identity())          # the full swing, on a posture 0.2 rad off and without the abduction
    out = flexion_two_band(base, alt, 0.75 * 30); mid = slice(70, 230)
    want = pose(0.3 + 0.5 * swing, abd)
    err = (R.from_rotvec(out[mid].reshape(-1, 3)).inv() * R.from_rotvec(want[mid].reshape(-1, 3))).magnitude()
    assert err.max() < 0.01                              # rad: posture and abduction from base, swing from alt
    assert np.abs(flexion_two_band(base, base, 22.5) - base).max() < 1e-6
    b2 = base.copy(); b2[100:110, 0] = np.nan; a2 = alt.copy(); a2[150:160, 1] = np.nan; o2 = flexion_two_band(b2, a2, 22.5)
    assert np.isnan(o2[100:110, 0]).all() and np.isfinite(o2[:100, 0]).all() and np.isfinite(o2[:, 1]).all()
    assert np.abs(o2[152:158, 1] - base[152:158, 1]).max() < 1e-6      # no second estimate there: the first one stands


def test_sight_line_is_the_camera_ray_in_the_world():
    T = 5; p = np.tile([0.3, -0.2, 4.0], (T, 1)); ray = p / np.linalg.norm(p, axis=1, keepdims=True); root_c = np.tile(R.from_euler("x", 5, degrees=True).as_rotvec(), (T, 1))
    assert np.allclose(sight_line_world(root_c, root_c, p), ray)                                    # the camera's frame is the world: the ray as it is
    turn = R.from_euler("y", 90, degrees=True); s = sight_line_world((turn * R.from_rotvec(root_c)).as_rotvec(), root_c, p)
    assert np.allclose(s, turn.apply(ray)) and np.allclose(np.linalg.norm(s, axis=1), 1.0)           # the world is the camera turned: so is the ray
    p[2] = np.nan; assert np.isnan(sight_line_world(root_c, root_c, p)[2]).all() and np.isfinite(sight_line_world(root_c, root_c, p)[[0, 1, 3, 4]]).all()


def test_fused_path_does_not_slide_when_the_share_changes():
    """A walker who stands still, in two views whose paths sit 1 m apart. The share of the root passes from one view
    to the other: a mean of the positions travels that metre; the fused steps do not move."""
    T = 120; pos = np.zeros((2, T, 3)); pos[1] += [1.0, 0.0, 0.0]; lifted = np.ones((2, T), bool)
    share = np.zeros((2, T)); share[0] = np.clip((80 - np.arange(T)) / 40.0, 0.0, 1.0); share[1] = 1.0 - share[0]
    mean = (share[:, :, None] * pos).sum(0); path = fuse_path(pos, share, lifted)
    assert np.ptp(mean[:, 0]) == 1.0 and np.ptp(path, axis=0).max() < 1e-9
    assert np.allclose(path[0], np.median(mean, axis=0))                                             # set on the mean position of the run


def test_fused_path_takes_each_views_step_across_its_line_of_sight():
    """A walk along a quarter circle. Each camera reads the step along its own line of sight at half its length.
    With the lines of sight the fused path is within 3 % of the true one; the share-weighted mean step is 25 % short."""
    T = 200; a = np.linspace(0.0, np.pi / 2, T); true = np.c_[4.0 * np.sin(a), np.full(T, 0.95), 4.0 * (1.0 - np.cos(a))]
    def seen(axis):
        d = np.diff(true, axis=0, prepend=true[:1]); d[:, axis] *= 0.5; return true[0] + np.cumsum(d, axis=0)
    pos = np.stack([seen(2), seen(0)]); sight = np.zeros((2, T, 3)); sight[0, :, 2] = 1.0; sight[1, :, 0] = 1.0   # camera 0 looks along z, camera 1 along x
    share = np.full((2, T), 0.5); lifted = np.ones((2, T), bool)
    def length(p):
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    fused = fuse_path(pos, share, lifted, sight, along=0.05); plain = fuse_path(pos, share, lifted)
    assert abs(length(fused) / length(true) - 1.0) < 0.03 and 0.72 < length(plain) / length(true) < 0.80
    sight[1] = np.nan; one = fuse_path(pos, share, lifted, sight, along=0.05)          # one line of sight unknown: that view counts the same in every direction
    assert np.isfinite(one).all() and length(fused) > length(one) > length(plain)


def test_fused_path_runs_and_single_view():
    """One view is its own path. A gap in every view starts a new run, and each run sits on the views' mean position."""
    rng = np.random.default_rng(3); T = 90; p = np.cumsum(rng.normal(0, 0.02, (T, 3)), axis=0)
    assert np.allclose(fuse_path(p[None], np.ones((1, T)), np.ones((1, T), bool)), p)
    pos = np.stack([p, p + [0.5, 0.0, 0.0]]); lifted = np.ones((2, T), bool); lifted[:, 40:50] = False; pos[:, 40:50] = np.nan; share = np.full((2, T), 0.5)
    out = fuse_path(pos, share, lifted)
    assert np.isnan(out[40:50]).all() and np.allclose(out[:40], p[:40] + [0.25, 0.0, 0.0]) and np.allclose(out[50:], p[50:] + [0.25, 0.0, 0.0])
    lifted[1, :20] = False; out = fuse_path(pos, np.where(lifted, share, 0.0) / np.maximum(np.where(lifted, share, 0.0).sum(0), 1e-9), lifted)
    assert np.isfinite(out[:40]).all() and np.allclose(np.diff(out[:40], axis=0), np.diff(p[:40], axis=0))      # a view joining adds no jump
