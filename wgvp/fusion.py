"""Rotation-space fusion of two lifted views: the library behind the merge stage (`wgvp.merge`) and the analysis
tool `tools/two_view_fusion.py`. Everything here is run-time: no ground truth is involved.

  usable_joints_rt   which SMPL-X body joints a view can vouch for in a frame (its 2D keypoints confident and inside the panel)
  hold_runs          hysteresis over consecutive frames (the shoulders' margin at the panel's top edge)
  ramp_inside        cross-fade weights at the edges of usable runs
  chordal_mean       weighted rotation average per (frame, joint)
  repair_root_glitches   a view's root flipping away and back within 0.2 s is interpolated across, before anything reads it
  fuse_root          the merged root orientation and each view's share of it (continuous in every term)
  smooth_rotations   short moving average in rotation space that never crosses a gap
  side_on_score      how side-on a camera sees the walker (1 = from the side, 0 = walking straight at it or away)
  flexion_two_band   a joint's flexion with its slow part (posture) from one estimate and its fast part (the stride's swing) from another
  export_mask        the frames an export keeps
  step_stats, geodesic, fit_yaw   measurements
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

LEGS = [0, 1, 3, 4, 6, 7]; ARMS = [12, 13, 15, 16, 17, 18, 19, 20]; SPINE = [2, 5, 8, 11, 14]
FLEX = [0, 1, 3, 4]                  # hips and knees: the joints whose flexion (the turn about their local x axis) carries the stride
TORSO_LEGS = sorted(LEGS + SPINE)   # hips, knees, ankles, feet, the three spine joints, neck and head: what an exported frame must have (an arm may be missing)
# SMPL-X body joint -> COCO-17 keypoints whose confidence says how well this joint's rotation is observed
# (the joint itself and the child segment it moves)
JOINT_KP = {0: [11, 13], 1: [12, 14], 2: [11, 12, 5, 6], 3: [13, 15], 4: [14, 16], 5: [5, 6, 11, 12], 6: [15], 7: [16], 8: [5, 6], 9: [15], 10: [16],
            11: [5, 6], 12: [5], 13: [6], 14: [5, 6], 15: [5, 7], 16: [6, 8], 17: [7, 9], 18: [8, 10], 19: [9], 20: [10]}
SHOULDER_KP = [5, 6]                 # COCO shoulders: at the panel's top edge only the head lies beyond them


REST_KP = [5, 6, 11, 12, 13, 14, 15, 16]     # shoulders, hips, knees, ankles: the keypoints a body at rest is checked on


def rest_held_seen(visible: np.ndarray, rest_held: np.ndarray, kp2d: np.ndarray, bbox: np.ndarray, min_conf: float = 0.5,
                   max_shift: float = 0.05) -> np.ndarray:
    """Which of a view's rest-held frames show the subject. A rest-held frame has no detection: the IMU labels the body
    at rest and the lifter was given the box of the neighbouring detection (session.rest_hold). The frame counts as seen
    when ViTPose, run inside that box, finds the shoulders, hips, knees and ankles where the nearest detected frame
    before or after has them: median confidence >= min_conf, median displacement <= max_shift of that frame's box
    height. A body at rest behind an occluder, or one that has left the box, fails both.
    visible, rest_held (T,) bool; kp2d (T, 17, 3); bbox (T, 4) xyxy, same pixels as kp2d. Returns (T,) bool."""
    visible = np.asarray(visible, bool); held = np.asarray(rest_held, bool) & ~visible; ok = np.zeros(len(visible), bool)
    vi = np.flatnonzero(visible); hi = np.flatnonzero(held)
    if len(vi) == 0 or len(hi) == 0:
        return ok
    kp = np.asarray(kp2d, float)[:, REST_KP]; nxt = np.searchsorted(vi, hi)
    with np.errstate(all="ignore"):
        conf_ok = np.nan_to_num(np.nanmedian(kp[hi, :, 2], axis=1), nan=0.0) >= min_conf
        for side in (nxt - 1, nxt):
            has = (side >= 0) & (side < len(vi)); a = vi[np.clip(side, 0, len(vi) - 1)]
            d = np.linalg.norm(kp[hi, :, :2] - kp[a, :, :2], axis=2) / np.maximum(bbox[a, 3] - bbox[a, 1], 1.0)[:, None]
            ok[hi] |= has & conf_ok & (np.nan_to_num(np.nanmedian(d, axis=1), nan=np.inf) <= max_shift)
    return ok


def seen_frames(z, k) -> np.ndarray:
    """The frames of one lifted view on which the subject is seen: detected (visible), or held at rest with keypoints
    that confirm it (rest_held_seen). z, k: the view's smpl_global.npz and kp2d_vitpose.npz (or dicts with the same
    fields); a lift from before the rest_held field has only the detected frames."""
    vis = np.asarray(z["visible"], bool)
    if "rest_held" not in (z.files if hasattr(z, "files") else z):
        return vis
    return vis | rest_held_seen(vis, z["rest_held"], k["kp2d"], k["bbox_xyxy"])


def hold_runs(loose: np.ndarray, strict: np.ndarray, breaks: np.ndarray | None = None) -> np.ndarray:
    """Hysteresis over consecutive frames: a run of frames that pass the loose test is kept when at least one of its
    frames passes the strict one. `breaks[i]` true cuts a run between frames i-1 and i (a jump in the frame index)."""
    loose = np.asarray(loose, bool); strict = np.asarray(strict, bool) & loose; out = np.zeros(len(loose), bool)
    if not loose.any():
        return out
    start = loose & ~np.r_[False, loose[:-1]]
    if breaks is not None:
        start |= loose & np.asarray(breaks, bool)
    run = np.cumsum(start); keep = np.zeros(int(run.max()) + 1, bool); keep[run[strict]] = True
    return loose & keep[run]


def usable_joints_rt(kp2d: np.ndarray, view_box, visible: np.ndarray, min_conf: float = 0.3, margin: float = 8.0,
                     shoulder_top_margin: float = 4.0, shoulder_top_hold: float = 2.0, frame_idx=None) -> np.ndarray:
    """Run-time test, no truth involved: SMPL-X body joint j is usable in a frame when the ViTPose keypoints it moves
    are confident and sit inside the view's panel with a margin, and the frame was not bridged. -> (T, 21) bool.
    The shoulders get a thinner margin at the panel's top edge: only the head lies beyond them there, ViTPose marks
    the top of the shoulder a few pixels above the joint, and against the samples' truth the trunk's error is the
    same with the shoulder keypoints 4 to 8 px below the edge as far from it (docs/design.md, 'The trunk at the top
    edge'); a hip keypoint at the top edge is a body cut at the waist, so every other keypoint keeps the full margin.
    That margin has hysteresis: a shoulder that has been `shoulder_top_margin` inside stays usable through the
    consecutive seen frames in which it is still `shoulder_top_hold` inside, so a keypoint that rides the margin
    (it moves by a few tenths of a pixel from one lift to the next) does not switch the trunk on and off frame by
    frame. `frame_idx` (the frames' indices, optional) cuts those runs where frames are missing."""
    conf = np.nan_to_num(kp2d[:, :, 2], nan=0.0); vis = np.asarray(visible, bool)
    if view_box is not None and len(view_box) == 4 and (view_box[2] - view_box[0]) > 0:
        x0, y0, x1, y1 = [float(b) for b in view_box]
        rest = (conf >= min_conf) & (kp2d[:, :, 0] > x0 + margin) & (kp2d[:, :, 0] < x1 - margin) & (kp2d[:, :, 1] < y1 - margin)
        top = np.full(kp2d.shape[1], float(margin)); top[SHOULDER_KP] = min(float(margin), float(shoulder_top_margin))
        ins = rest & (kp2d[:, :, 1] > y0 + top[None])
        hold = min(float(shoulder_top_hold), float(top[SHOULDER_KP[0]]))
        if hold < top[SHOULDER_KP[0]]:
            breaks = None
            if frame_idx is not None and len(frame_idx) > 1:
                d = np.diff(np.asarray(frame_idx, float)); breaks = np.r_[False, d > np.median(d)]
            for c in SHOULDER_KP:
                ins[:, c] = hold_runs(rest[:, c] & (kp2d[:, c, 1] > y0 + hold) & vis, ins[:, c], breaks)
    else:
        ins = conf >= min_conf
    return np.stack([ins[:, JOINT_KP[j]].all(1) for j in range(21)], 1) & vis[:, None]


def ramp_inside(mask: np.ndarray, k: int) -> np.ndarray:
    """(T, J) bool -> (T, J) float weights that rise linearly from 0 to 1 over the first k frames of every run of
    True and fall over the last k, and are exactly 0 where the mask is False. Used to cross-fade between views at
    the edges of their usable segments; where only one view is usable the weights renormalise to that view and the
    ramp has no effect."""
    m = np.asarray(mask, bool); T, J = m.shape; w = np.zeros((T, J), float); k = max(int(k), 1)
    for j in range(J):
        edges = np.flatnonzero(np.diff(np.r_[0, m[:, j].astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            n = b - a; i = np.arange(n)
            w[a:b, j] = np.minimum(1.0, np.minimum((i + 1) / k, (n - i) / k))
    return w


def smooth_rotations(aa: np.ndarray, window: int) -> np.ndarray:
    """Moving average of rotations over time, (T, J, 3) axis-angle -> same, computed as the normalised mean of unit
    quaternions (hemisphere-aligned to the window's centre) over `window` frames, within runs of emitted frames
    only: NaN gaps stay gaps and are not averaged across. Near a run's end the window shrinks on BOTH sides, so
    the average stays centred on its frame (a one-sided window would pull a moving joint toward the run's inside).
    window <= 1 returns the input."""
    if window <= 1:
        return aa
    T, J = aa.shape[:2]; out = np.full_like(aa, np.nan); h = window // 2
    for j in range(J):
        ok = np.isfinite(aa[:, j]).all(1)
        if not ok.any():
            continue
        q = np.full((T, 4), np.nan); q[ok] = R.from_rotvec(aa[ok, j]).as_quat()
        edges = np.flatnonzero(np.diff(np.r_[False, ok, False].astype(int)))
        for a, b in zip(edges[::2], edges[1::2]):                      # stay inside this run of emitted frames
            for t in range(a, b):
                n = min(h, t - a, b - 1 - t); qs = q[t - n:t + n + 1]; qs = qs * np.sign(qs @ q[t] + 1e-12)[:, None]
                out[t, j] = R.from_quat(qs.mean(0) / np.linalg.norm(qs.mean(0))).as_rotvec()
    return out


def repair_root_glitches(root_aa: np.ndarray, fps: float = 30.0, max_s: float = 0.2, jump_deg: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """A monocular lift can take the front/back ambiguity for an instant: its root steps by more than jump_deg
    between two consecutive frames and, within max_s, is back within jump_deg / 2 of where it was. No body turns
    like that; the view's joints are unaffected (measured on the rendered session where it happens: the leg error
    of the two flipped frames is that of their neighbours), and left in, the two steps read as a snap to fuse_root's
    offline rule, which then sets the wrong view aside for the whole turn. Those frames get the root interpolated
    across them (slerp between the frames either side). A step that does not come back is left alone: that is a
    snap, and fuse_root's business. (T, 3) axis-angle with NaN gaps -> the repaired roots and the (T,) mask of
    repaired frames."""
    root_aa = np.asarray(root_aa, float); T = len(root_aa); out = root_aa.copy(); fixed = np.zeros(T, bool)
    ok = np.isfinite(root_aa).all(1); G = max(1, int(round(max_s * fps)))
    if ok.sum() < 3:
        return out, fixed
    Q = np.full((T, 4), np.nan); Q[ok] = R.from_rotvec(root_aa[ok]).as_quat()
    def ang(a, b): return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))
    t = 1
    while t < T:
        if ok[t] and ok[t - 1] and ang(Q[t], Q[t - 1]) > jump_deg:
            back = next((k for k in range(1, G + 1) if t + k < T and ok[t + k] and ang(Q[t + k], Q[t - 1]) < 0.5 * jump_deg), None)
            if back is not None:
                q0 = Q[t - 1]; q1 = Q[t + back] * (1.0 if Q[t + back] @ q0 >= 0 else -1.0)
                for j in range(back):
                    if ok[t + j]:
                        m = q0 + (q1 - q0) * ((j + 1) / (back + 1.0)); out[t + j] = R.from_quat(m / np.linalg.norm(m)).as_rotvec(); fixed[t + j] = True
                t += back + 1; continue
        t += 1
    return out, fixed


def fuse_root(roots: list[np.ndarray], usable: list[np.ndarray], lifted: list[np.ndarray], ramp: int = 15,
              soft_lo: float = 30.0, soft_hi: float = 60.0, n_joints: int = 21, return_raw: bool = False, cont_scale: float = 30.0,
              human_turn_deg_s: float = 150.0, snap_scale_deg_s: float = 150.0, fps: float = 30.0, snap_deg: float = 90.0,
              snap_window_s: float = 0.25, max_turn_s: float = 4.0, arb_smooth: bool = True):
    """Root orientation of the merge and each view's share of it. Inputs per view, all in one world frame (per-view
    heading offsets already removed): roots (T, 3), usable (T, J) bool, lifted (T,) bool. Returns the fused root
    (T, 3) axis-angle, NaN where no view lifted the frame, and the normalised weights (V, T); with return_raw also
    the weights before normalisation (V, T), each in [0, 1].

    Each view's root carries the weight  presence x (usable fraction)^2: presence ramps from 0 to 1 over the first
    and last `ramp` frames of every lifted run, so a view that enters or leaves cannot step the root, and the usable
    fraction is averaged over `ramp` frames, so joints flickering in and out of a view cannot either; a view that
    sees 5 of 21 joints carries 6 % of the weight of one that sees them all. When the roots disagree by more than
    soft_lo degrees one of them is wrong (two good lifts of the same walker agree to within ~10 deg), so the weights
    are sharpened towards the heaviest view, fully by soft_hi; the sharpening is a continuous function of the weights
    and the disagreement, so nothing here introduces a discontinuity that is not in the inputs. When the views
    disagree by more than soft_lo, two continuity terms arbitrate before the sharpening (a monocular lifter takes a
    180-degree turn the wrong way round in one view by snapping through the front/back ambiguity, after which the
    two headings are far apart and their mean is meaningless): a view's weight is divided by 1 + g ((its own turning
    rate since its previous frame - human_turn_deg_s)+ / snap_scale_deg_s)^2, which discounts the snap itself (a
    person turns at up to about 150 deg/s, a snapping lift at 500 or more) and leaves a turn at a human rate alone,
    and by 1 + g (its angle to the previous fused output / cont_scale)^2, which keeps the output with the view it was
    following until the views agree again. g is the size of the disagreement on the same scale as the sharpening
    (0 at soft_lo, 1 by soft_hi), so the two terms fade in with it: switched on at full strength the moment the
    disagreement crossed soft_lo, they halved a view's weight from one frame to the next whenever two lifts sat
    about 30 deg apart (every turn), and the fused heading hopped by several degrees each time (measured on the
    rendered sessions: the heading's frame-to-frame jitter in turns was 2 to 5 times the truth's). The weighted mean
    is taken on unit quaternions, hemisphere-aligned to the heaviest view.

    Those two terms are causal and cannot settle the other form of the same failure: a view that turns the wrong
    way round at a human rate, indistinguishable from the right one while both turn, and gives itself away only by
    snapping onto the true heading when the turn is over (by then the merge had locked onto it if it was the
    heavier view). The merge runs offline, so a third rule looks at the whole recording first: within every
    disagreement episode (frames with two views more than soft_lo apart, gaps up to `ramp` frames bridged), a view
    whose own root turns by more than snap_deg within snap_window_s, during the episode or just after it, is the
    one that took the turn the wrong way (a person cannot turn 90 degrees in a quarter of a second; the snap is the
    lift's own correction, so that view was wrong on one side of it, inside the episode), and it carries no weight
    over the episode, up to max_turn_s either side of its snap (its weight fades out over `ramp` frames before the
    episode and back in after it, so the hand-over cannot step the root). The same episode can also end without a snap when the
    wrong view's coverage simply runs out (the walker leaves its frame mid-turn): the disagreement is never resolved,
    and at the hand-off the merged root would jump onto the surviving view. A view whose coverage ends, or begins,
    inside an episode while another view continues through it is treated like a snapping one and carries no weight
    over the episode. When no view snaps or departs, or more than one does, the rule is silent and the causal terms
    decide.

    The causal terms feed on their own output (the continuity term measures each view against the previous fused
    root), so their verdict can flip from one frame to the next while two views sit near soft_lo apart. A second
    pass (arb_smooth) therefore averages what the arbitration did to each view's weight over `ramp` frames and
    recomputes the shares and the root from that. Measured on the six rendered sessions (24 turns), with the faded
    drop: heading jitter inside turns 5.5 times the truth's before and 0.9 after, peak turning rate 2.9 times the
    truth's before and 1.3 after.

    The share is also the right weight for the view's body joints: a lifter that got the root's pitch wrong (the
    along view of the sample at close range: +40 deg) has hips that compensate for it, and the rest of its body
    solved in the wrong gravity frame."""
    V = len(roots); T = roots[0].shape[0]
    L = np.stack([np.asarray(l, bool) for l in lifted])                                                     # (V, T)
    pres = np.stack([ramp_inside(L[i][:, None], ramp)[:, 0] for i in range(V)])
    frac = np.stack([_moving_average(np.asarray(u, bool).sum(1) / float(n_joints), ramp) for u in usable])
    w = pres * frac ** 2; w[~L] = 0.0
    Q = np.stack([np.where(np.isfinite(r).all(1)[:, None], R.from_rotvec(np.nan_to_num(r)).as_quat(), np.nan) for r in roots])
    w = _drop_snapping_views(w, Q, L, soft_lo, ramp, snap_deg, snap_window_s, max_turn_s, fps)
    out = np.full((T, 3), np.nan); share = np.zeros((V, T)); arb = np.ones((V, T)); prev = None; gap = 0
    last_q = [None] * V; last_t = np.full(V, -10 ** 9); r0 = human_turn_deg_s / max(fps, 1e-6); rs = snap_scale_deg_s / max(fps, 1e-6)   # deg per frame
    def ang(a, b): return np.degrees(2.0 * np.arccos(np.clip(np.abs(a @ b), 0.0, 1.0)))
    for t in range(T):
        idx = np.flatnonzero(L[:, t] & np.isfinite(Q[:, t]).all(1))
        if not len(idx):
            gap += 1; continue
        ww = w[idx, t].copy()
        if ww.sum() <= 1e-9:                                   # lifted but nothing usable in any view (bridged frames): plain mean
            ww[:] = 1.0
        ww0 = ww.copy()
        k = int(np.argmax(ww)); q = Q[idx, t].copy(); q *= np.sign(q @ q[k] + 1e-12)[:, None]
        d = ang(q, q[k])                                                                                     # each root's angle to the heaviest
        g = float(np.clip((d.max() - soft_lo) / (soft_hi - soft_lo), 0.0, 1.0)) if len(idx) > 1 else 0.0     # how much of a disagreement this is: 0 below soft_lo, 1 by soft_hi
        if g > 0.0:                                                                                         # disagreement: continuity arbitrates
            rate = np.array([ang(Q[i, t], last_q[i]) / max(t - last_t[i], 1) if (last_q[i] is not None and t - last_t[i] <= ramp) else 0.0 for i in idx])
            ww = ww / (1.0 + g * (np.maximum(rate - r0, 0.0) / rs) ** 2)
            if prev is not None and gap <= ramp:
                ww = ww / (1.0 + g * (ang(q, prev) / cont_scale) ** 2)
            k = int(np.argmax(ww)); q *= np.sign(q @ q[k] + 1e-12)[:, None]; d = ang(q, q[k])
        p = 8.0 * np.clip((d - soft_lo) / (soft_hi - soft_lo), 0.0, 1.0)                                    # 0 below soft_lo, 8 at soft_hi
        ww = ww * (ww / ww[k]) ** p; share[idx, t] = ww / ww.sum()
        a = np.where(ww0 > 0, ww / np.maximum(ww0, 1e-12), 0.0); arb[idx, t] = a / max(float(a.max()), 1e-12)   # what the arbitration made of each view's weight (the preferred view: 1)
        m = (ww[:, None] * q).sum(0); m /= np.linalg.norm(m); out[t] = R.from_quat(m).as_rotvec(); prev = m; gap = 0
        for i in idx:
            last_q[i] = Q[i, t]; last_t[i] = t
    if arb_smooth and ramp > 1:                                                                            # second pass: the arbitration, averaged over the cross-fade time
        arb_s = np.stack([_moving_average(arb[i], ramp) for i in range(V)])
        for t in range(T):
            idx = np.flatnonzero(L[:, t] & np.isfinite(Q[:, t]).all(1))
            if not len(idx):
                continue
            ww = w[idx, t] * arb_s[idx, t]
            if ww.sum() <= 1e-9:
                ww = np.ones(len(idx))
            k = int(np.argmax(ww)); q = Q[idx, t].copy(); q *= np.sign(q @ q[k] + 1e-12)[:, None]
            share[:, t] = 0.0; share[idx, t] = ww / ww.sum()
            m = (ww[:, None] * q).sum(0); m /= np.linalg.norm(m); out[t] = R.from_quat(m).as_rotvec()
    return (out, share, w) if return_raw else (out, share)


def _drop_snapping_views(w: np.ndarray, Q: np.ndarray, L: np.ndarray, soft_lo: float, ramp: int, snap_deg: float,
                         snap_window_s: float, max_turn_s: float, fps: float) -> np.ndarray:
    """fuse_root's offline rule (see there). w, L (V, T); Q (V, T, 4) quaternions, NaN where absent. Returns w with
    the view that snaps, or whose coverage ends or begins, inside a disagreement episode zeroed over that episode,
    faded out over `ramp` frames before it and back in over `ramp` frames after it."""
    V, T = w.shape; win = max(1, int(round(snap_window_s * fps))); mx = max(win, int(round(max_turn_s * fps)))
    ok = L & np.isfinite(Q).all(2)
    def ang(a, b): return np.degrees(2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)))
    def runs(mask):
        e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0])); return list(zip(e[::2], e[1::2]))        # [a, b) pairs
    dis = np.zeros(T, bool)
    for i in range(V):
        for j in range(i + 1, V):
            both = ok[i] & ok[j]
            if both.any():
                a = np.zeros(T); a[both] = ang(Q[i][both], Q[j][both]); dis |= both & (a > soft_lo)
    if not dis.any():
        return w
    episodes = []
    for a_, b_ in runs(dis):
        if episodes and a_ - episodes[-1][1] <= ramp:
            episodes[-1][1] = b_
        else:
            episodes.append([a_, b_])
    snaps = []                                                                                           # (view, first frame, last frame)
    for i in range(V):
        s = np.zeros(T, bool); t = np.arange(win, T); valid = ok[i, t] & ok[i, t - win]; tt = t[valid]
        if len(tt):
            s[tt] = ang(Q[i, tt], Q[i, tt - win]) > snap_deg
        snaps += [(i, a_ - win, b_ - 1) for a_, b_ in runs(s)]
    for i in range(V):                                                                               # coverage ending or beginning while another view continues
        others = ok[np.arange(V) != i]
        for a_, b_ in runs(ok[i]):
            if b_ < T and others[:, b_:min(b_ + win, T)].any():
                snaps.append((i, b_ - 1, b_ - 1))
            if a_ > 0 and others[:, max(a_ - win, 0):a_].any():
                snaps.append((i, a_, a_))
    w = w.copy()
    for a_, b_ in episodes:
        hits = [(i, s0, s1) for i, s0, s1 in snaps if s0 <= b_ + win and s1 >= a_ - 1]
        if len({i for i, _, _ in hits}) != 1:
            continue
        i = hits[0][0]; s0 = min(h[1] for h in hits); s1 = max(h[2] for h in hits)
        lo = max(a_, s0 - mx); hi = min(max(b_, s1 + 1), s1 + mx, T)
        f = np.ones(T); f[lo:hi] = 0.0                                                                   # out over the episode, faded out before it and back in after it
        pre = np.arange(max(lo - ramp, 0), lo); f[pre] = (lo - pre) / (ramp + 1.0)
        post = np.arange(hi, min(hi + ramp, T)); f[post] = (post - hi + 1) / (ramp + 1.0)
        w[i] *= f
    return w


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(x, float)
    c = np.cumsum(np.r_[0.0, x]); h = k // 2; i = np.arange(len(x)); a_ = np.clip(i - h, 0, len(x)); b_ = np.clip(i + h + 1, 0, len(x))
    return (c[b_] - c[a_]) / (b_ - a_)


def moving_average_in_runs(x: np.ndarray, k: int, centred: bool = False) -> np.ndarray:
    """(T,) or (T, D) with NaN gaps -> moving average over k frames inside each run of finite rows; gaps stay NaN.
    Near a run's end the window is cut on the outer side only (right for a slow trend: the average keeps its
    length), or with `centred` on both sides (right for a position: a one-sided window pulls a moving point toward
    the run's inside and halves the last steps of a path)."""
    x = np.asarray(x, float); one = x.ndim == 1; X = x[:, None] if one else x; out = np.full_like(X, np.nan)
    ok = np.isfinite(X).all(1); edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        i = np.arange(b - a); h = np.minimum(k // 2, np.minimum(i, b - a - 1 - i))
        for d in range(X.shape[1]):
            if centred and k > 1:
                c = np.cumsum(np.r_[0.0, X[a:b, d]]); out[a:b, d] = (c[i + h + 1] - c[i - h]) / (2 * h + 1)
            else:
                out[a:b, d] = _moving_average(X[a:b, d], k)
    return out[:, 0] if one else out


def export_mask(merged_out: np.ndarray, required=TORSO_LEGS, min_run: int = 15) -> np.ndarray:
    """Frames an export keeps: every required joint is output (seen by at least one view), in runs of at least
    min_run frames. The default asks for the legs and the torso: a frame with a leg or a trunk joint that no view
    saw is dropped whole, a frame with only an arm missing is kept (the arm's joints stay NaN with weight 0).
    (T, 21, 3) with NaN for joints not output -> (T,) bool."""
    ok = np.isfinite(merged_out[:, list(required)]).all(axis=(1, 2))
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a_, b_ in zip(edges[::2], edges[1::2]):
        if b_ - a_ < min_run:
            ok[a_:b_] = False
    return ok


def step_stats(aa: np.ndarray) -> dict:
    """Frame-to-frame continuity of an estimate (T, 21, 3): largest per-joint rotation step between consecutive
    frames where both are emitted; p99 and the share of steps over 20 deg."""
    ok = np.isfinite(aa).all(2)                                          # (T, 21)
    both = ok[1:] & ok[:-1]
    if not both.any():
        return {"p99_deg": float("nan"), "frac_over_20": float("nan"), "n": 0}
    a0 = aa[:-1][both]; a1 = aa[1:][both]
    d = np.degrees((R.from_rotvec(a0).inv() * R.from_rotvec(a1)).magnitude())
    return {"p99_deg": float(np.percentile(d, 99)), "frac_over_20": float((d > 20).mean()), "n": int(len(d))}


def geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (..., 3) axis-angle -> degrees; NaN rotations (no output) give NaN."""
    sh = a.shape[:-1]; a2 = a.reshape(-1, 3); b2 = b.reshape(-1, 3); ok = np.isfinite(a2).all(1) & np.isfinite(b2).all(1); out = np.full(len(a2), np.nan)
    out[ok] = np.degrees((R.from_rotvec(a2[ok]).inv() * R.from_rotvec(b2[ok])).magnitude())
    return out.reshape(sh)


def side_on_score(root_cam_aa: np.ndarray, transl_cam: np.ndarray) -> np.ndarray:
    """How side-on a camera sees the body, per frame: the squared cosine between the pelvis's left-right axis and the
    ray from the camera to the pelvis, both in the camera's frame (smpl_incam.npz: global_orient (T,3), transl (T,3)).
    1 = the camera looks along that axis, so the hips and knees flex in the image plane; 0 = the body walks straight
    at the camera or away from it, and their swing runs along the line of sight, where one camera reads it worst.
    A mirrored picture gives the same score. -> (T,), NaN where an input is NaN."""
    a = np.asarray(root_cam_aa, float); p = np.asarray(transl_cam, float); out = np.full(len(a), np.nan)
    ok = np.isfinite(a).all(1) & np.isfinite(p).all(1); ok[ok] = np.linalg.norm(p[ok], axis=1) > 1e-6
    if ok.any():
        lat = R.from_rotvec(a[ok]).apply([1.0, 0.0, 0.0]); ray = p[ok] / np.linalg.norm(p[ok], axis=1, keepdims=True); out[ok] = (lat * ray).sum(1) ** 2
    return out


def sight_line_world(root_world_aa: np.ndarray, root_cam_aa: np.ndarray, transl_cam: np.ndarray) -> np.ndarray:
    """The camera's line of sight to the pelvis in the lifter's world frame, per frame: the unit ray from the camera to
    the pelvis in the camera's frame (smpl_incam.npz transl), turned by the rotation that takes the camera-frame root
    to the world-frame root. -> (T, 3) unit vectors, NaN where an input is NaN."""
    a = np.asarray(root_world_aa, float); c = np.asarray(root_cam_aa, float); p = np.asarray(transl_cam, float); out = np.full((len(a), 3), np.nan)
    ok = np.isfinite(a).all(1) & np.isfinite(c).all(1) & np.isfinite(p).all(1); ok[ok] = np.linalg.norm(p[ok], axis=1) > 1e-6
    if ok.any():
        out[ok] = (R.from_rotvec(a[ok]) * R.from_rotvec(c[ok]).inv()).apply(p[ok] / np.linalg.norm(p[ok], axis=1, keepdims=True))
    return out


def fuse_path(pos: np.ndarray, share: np.ndarray, lifted: np.ndarray, sight: np.ndarray | None = None, along: float = 0.05) -> np.ndarray:
    """One pelvis path from the views' aligned paths (pos (V, T, 3), y up; share (V, T); lifted (V, T) bool).

    The views' per-frame STEPS are fused and summed, not their positions: two monocular paths sit decimetres to
    metres apart while the walker moves, so a share-weighted mean of positions slides whenever the shares change.
    Each view's horizontal step counts in full across its camera's line of sight (`sight`, (V, T, 3) in the same
    world; NaN or None where unknown: that view's step then counts the same in every direction) and `along` of
    that along it, where the step is a change of monocular depth. The vertical step is the share-weighted mean.
    Every run of frames is then set on the share-weighted mean position by the median offset, so the path keeps
    the views' origin and none of their drift against each other.

    -> (T, 3), NaN where no view is lifted."""
    pos = np.asarray(pos, float); V, T, _ = pos.shape; share = np.asarray(share, float); along = max(float(along), 1e-3)
    fin = np.asarray(lifted, bool) & np.isfinite(pos).all(2); some = fin.any(0); out = np.full((T, 3), np.nan)
    wa = np.where(fin, share, 0.0); none = some & (wa.sum(0) <= 0); wa[:, none] = fin[:, none]                     # no share on a lifted frame: the plain mean
    anchor = (wa[:, :, None] * np.nan_to_num(pos)).sum(0) / np.maximum(wa.sum(0), 1e-9)[:, None]
    has = np.zeros((V, T), bool); has[:, 1:] = fin[:, 1:] & fin[:, :-1]; step = np.zeros((V, T, 3)); step[:, 1:] = np.nan_to_num(pos[:, 1:] - pos[:, :-1]); step[~has] = 0.0
    w = np.where(has, share, 0.0); ws = w.sum(0); v = (w[:, :, None] * step).sum(0) / np.maximum(ws, 1e-9)[:, None]
    if sight is not None:
        A = np.zeros((T, 2, 2)); b = np.zeros((T, 2))
        for i in range(V):
            r = np.nan_to_num(np.asarray(sight[i], float)[:, [0, 2]]); n = np.linalg.norm(r, axis=1); known = n > 1e-6; r[known] /= n[known][:, None]
            W = w[i][:, None, None] * (np.eye(2)[None] - (1.0 - along) * known[:, None, None] * r[:, :, None] * r[:, None, :])
            A += W; b += np.einsum("tij,tj->ti", W, step[i][:, [0, 2]])
        ok = ws > 0; hz = v[:, [0, 2]]; hz[ok] = np.linalg.solve(A[ok], b[ok][:, :, None])[:, :, 0]; v[:, [0, 2]] = hz
    start = some & ~(ws > 0); run = np.cumsum(start)                  # a run begins where no view steps into the frame
    for k in range(1, int(run.max()) + 1 if T else 1):
        m = some & (run == k); p = np.cumsum(np.where(start[m][:, None], 0.0, v[m]), axis=0); out[m] = p + np.median(anchor[m] - p, axis=0)
    return out


def _twist_x(aa: np.ndarray):
    """(N, 3) axis-angle -> (the part of each rotation that turns about the x axis, its angle in radians): the
    swing-twist split r = swing * twist. NaN rows count as no rotation."""
    q = R.from_rotvec(np.nan_to_num(aa)).as_quat(); tw = np.zeros_like(q); tw[:, 0] = q[:, 0]; tw[:, 3] = q[:, 3]
    n = np.linalg.norm(tw, axis=1); bad = n < 1e-8; tw[bad] = [0.0, 0.0, 0.0, 1.0]; n[bad] = 1.0; tw = tw / n[:, None]
    tw = tw * np.where(tw[:, 3:4] < 0, -1.0, 1.0)
    return R.from_quat(tw), 2.0 * np.arctan2(tw[:, 0], tw[:, 3])


def _gauss_lowpass(x: np.ndarray, sigma: float) -> np.ndarray:
    """(T, D) with NaN -> Gaussian low-pass along time, a normalised convolution over the finite samples (the
    window shrinks at a gap or an end instead of reading across it as zero); NaN stays NaN."""
    h = max(1, int(np.ceil(3.0 * sigma))); g = np.exp(-0.5 * (np.arange(-h, h + 1) / max(sigma, 1e-6)) ** 2); ok = np.isfinite(x)
    num = np.stack([np.convolve(np.where(ok[:, j], x[:, j], 0.0), g, "same") for j in range(x.shape[1])], 1)
    den = np.stack([np.convolve(ok[:, j].astype(float), g, "same") for j in range(x.shape[1])], 1)
    return np.where(ok, num / np.maximum(den, 1e-9), np.nan)


def flexion_two_band(base: np.ndarray, alt: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Two estimates (T, J, 3) of the same joints -> `base` with the fast part of its flexion taken from `alt`.
    Each rotation is split into the flexion about the joint's local x axis and the rest (r = rest * flexion). The
    output keeps base's rest and the slow part of base's flexion angle (the posture: what a Gaussian low-pass of
    sigma_frames passes) and takes everything faster (the stride's swing) from alt:
    angle = base + (d - lowpass(d)), d = alt - base. NaN where base is NaN; where alt is NaN the output is base."""
    sh = base.shape; T = sh[0]; b = np.asarray(base, float).reshape(-1, 3); a = np.asarray(alt, float).reshape(-1, 3)
    fin = np.isfinite(b).all(1); both = fin & np.isfinite(a).all(1)
    tw_b, th_b = _twist_x(b); rest = R.from_rotvec(np.nan_to_num(b)) * tw_b.inv(); _, th_a = _twist_x(a)
    d = np.where(both, (th_a - th_b + np.pi) % (2 * np.pi) - np.pi, np.nan).reshape(T, -1)
    th = th_b + np.nan_to_num(d - _gauss_lowpass(d, sigma_frames)).ravel()
    out = (rest * R.from_rotvec(np.c_[th, np.zeros_like(th), np.zeros_like(th)])).as_rotvec(); out[~fin] = np.nan
    return out.reshape(sh).astype(np.asarray(base).dtype)


def chordal_mean(aa_list: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Weighted rotation average per (frame, joint): aa_list = [(T,J,3)] per view, w (V,T,J). Quaternion method
    (largest eigenvector of the weighted outer-product sum), exact for two rotations."""
    V = len(aa_list); T, J = aa_list[0].shape[:2]
    Q = np.stack([R.from_rotvec(np.nan_to_num(a.reshape(-1, 3))).as_quat().reshape(T, J, 4) for a in aa_list])     # (V,T,J,4)
    out = np.zeros((T, J, 3), np.float32)
    for t in range(T):
        for j in range(J):
            ww = w[:, t, j]
            if ww.sum() <= 0:
                ww = np.ones(V)
            q = Q[:, t, j]; q = q * np.sign(np.sum(q * q[:1], axis=1, keepdims=True) + 1e-12)   # same hemisphere as view 0
            M = sum(wi * np.outer(qi, qi) for wi, qi in zip(ww, q))
            vals, vecs = np.linalg.eigh(M); out[t, j] = R.from_quat(vecs[:, -1]).as_rotvec()
    return out


# SMPL-X body joints 0-21 of the neutral mean shape, rest pose, relative to the pelvis (metres; from the SMPL-X neutral
# model: smplx.create(..., model_type="smplx").joints with zero pose and shape), and their parents. Enough for a
# model-free forward kinematics of the merged pose when only joint POSITIONS of the mean body are needed (the feet
# for the walkway calibration); the real body shape moves them by a few centimetres.
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPLX_REST_JOINTS = np.array([[0.0, 0.0, 0.0], [0.0582, -0.0928, -0.026], [-0.0633, -0.1039, -0.0213], [-0.0028, 0.1099, -0.0276], [0.1129, -0.4715, -0.0354],
                              [-0.1075, -0.4663, -0.0381], [0.0067, 0.2417, -0.0336], [0.0694, -0.8746, -0.0673], [-0.0921, -0.877, -0.0583], [-0.0046, 0.294, -0.0051],
                              [0.1167, -0.9326, 0.0509], [-0.1309, -0.9353, 0.0608], [-0.0168, 0.4591, -0.0367], [0.0417, 0.3789, -0.0123], [-0.0523, 0.3783, -0.0185],
                              [0.008, 0.6196, -0.016], [0.161, 0.4367, -0.0278], [-0.1549, 0.4318, -0.0312], [0.4151, 0.3645, -0.0703], [-0.4261, 0.3953, -0.0576],
                              [0.6671, 0.3877, -0.0727], [-0.6753, 0.3908, -0.073]])
L_FOOT, R_FOOT, L_ANKLE, R_ANKLE = 10, 11, 7, 8


def fk_joints(root_aa: np.ndarray, body_aa: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """World positions (T, 22, 3) of the SMPL-X body joints of the mean shape for root (T,3), body pose (T,21,3) and
    translation (T,3); the pelvis sits at the translation. Rows with a NaN root or translation are NaN; a NaN body
    joint is taken as identity (its children then hang in the rest direction of their parent)."""
    T = root_aa.shape[0]; out = np.full((T, 22, 3), np.nan)
    ok = np.isfinite(root_aa).all(1) & np.isfinite(transl).all(1)
    if not ok.any():
        return out
    Rg = [None] * 22; pos = [None] * 22
    with np.errstate(all="ignore"):
        Rg[0] = R.from_rotvec(root_aa[ok]); pos[0] = transl[ok].astype(float)
        for j in range(1, 22):
            par = SMPLX_PARENTS[j]; loc = np.nan_to_num(body_aa[ok, j - 1])
            pos[j] = pos[par] + Rg[par].apply(SMPLX_REST_JOINTS[j] - SMPLX_REST_JOINTS[par]); Rg[j] = Rg[par] * R.from_rotvec(loc)
    out[ok] = np.stack(pos, axis=1)
    return out


# left <-> right partner of every SMPL-X body joint (0-based body index: joint 1..21 of the kinematic tree)
MIRROR_SWAP = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 18, 17, 20, 19]
_MIRROR_SIGN = np.array([1.0, -1.0, -1.0])   # a rotation conjugated by the reflection x -> -x: rotvec (x, y, z) -> (x, -y, -z)


def mirror_body(body_aa: np.ndarray) -> np.ndarray:
    """The body pose of the mirror image of a person: left and right joints swapped and every local rotation
    conjugated by the sagittal reflection. (T, 21, 3) -> (T, 21, 3). The map is an involution and exact for the
    bilaterally symmetric SMPL-X template; it holds for ANY vertical mirror plane, since two such reflections
    differ by a yaw and a horizontal shift, which the view alignment absorbs."""
    return np.asarray(body_aa)[:, MIRROR_SWAP] * _MIRROR_SIGN


def mirror_root(root_aa: np.ndarray) -> np.ndarray:
    """Root orientation of the mirror image (reflection about the world plane x = 0). (T, 3) -> (T, 3)."""
    return np.asarray(root_aa) * _MIRROR_SIGN


def mirror_transl(transl: np.ndarray) -> np.ndarray:
    return np.asarray(transl) * np.array([-1.0, 1.0, 1.0])


def mirror_check(body_ref: np.ndarray, body_v: np.ndarray, both_ok: np.ndarray, min_asym_deg: float = 15.0) -> dict:
    """Is `body_v` the mirror image of `body_ref`? On the frames `both_ok` (T,) where the reference's two knees
    differ in flexion by more than `min_asym_deg` (one leg swinging), the median geodesic distance of the KNEES as
    they are and after mirroring `body_v`. Walking makes the test sharp there: the knees are half a cycle apart, so
    a swapped pair disagrees by the flexion range. Measured on six rendered two-camera sessions: as-is 11-16 deg vs
    mirrored 21-40; on a mirrored panel of a real clip: 44 vs 8. A view counts as mirrored when the mirrored
    distance is under 0.6 of the as-is one and at least 6 deg smaller. Returns the two distances, the frames used
    and the verdict (False when there are too few frames to tell)."""
    ok = np.asarray(both_ok, bool) & np.isfinite(np.asarray(body_ref)[:, [3, 4]]).all((1, 2)) & np.isfinite(np.asarray(body_v)[:, [3, 4]]).all((1, 2))
    kn = np.degrees(np.linalg.norm(np.nan_to_num(np.asarray(body_ref)[:, [3, 4]]), axis=2)); swing = ok & (np.abs(kn[:, 0] - kn[:, 1]) > min_asym_deg)
    idx = np.flatnonzero(swing)
    if len(idx) < 30:
        return {"mirrored": False, "n_frames": int(len(idx)), "as_is_deg": None, "mirrored_deg": None, "note": "fewer than 30 frames with one knee swinging in both views"}
    a = np.asarray(body_ref)[idx][:, [3, 4]]; b = np.asarray(body_v)[idx][:, [3, 4]]; bm = mirror_body(np.asarray(body_v)[idx])[:, [3, 4]]
    d0 = float(np.nanmedian(geodesic(a, b))); d1 = float(np.nanmedian(geodesic(a, bm)))
    return {"mirrored": bool(d1 < 0.6 * d0 and d0 - d1 >= 6.0), "n_frames": int(len(idx)), "as_is_deg": d0, "mirrored_deg": d1}


def collapsed_legs(kp: np.ndarray, rel: float = 0.08, rel_knee: float = 0.12, min_conf: float = 0.3) -> np.ndarray:
    """Frames where a 2D pose estimator put BOTH legs on one leg: the two ankles within `rel` of the shank length
    of each other and the knees within `rel_knee` (COCO-17 keypoints (T, 17, 3)). A side-on walker's legs cross
    once a stride, which also triggers this for a frame or two; the 3D lifter copes either way (measured: leg error
    against the truth 8-16 deg on these frames vs 13 deg elsewhere), so this is for display, not for weighting."""
    kp = np.asarray(kp, float); ank = np.linalg.norm(kp[:, 15, :2] - kp[:, 16, :2], axis=1); kne = np.linalg.norm(kp[:, 13, :2] - kp[:, 14, :2], axis=1)
    shank = np.maximum(np.linalg.norm(kp[:, 13, :2] - kp[:, 15, :2], axis=1), np.linalg.norm(kp[:, 14, :2] - kp[:, 16, :2], axis=1))
    conf = np.nan_to_num(kp[:, [13, 14, 15, 16], 2]).min(1)
    with np.errstate(invalid="ignore"):
        return (ank < rel * shank) & (kne < rel_knee * shank) & (conf >= min_conf) & np.isfinite(shank)


def heading_yaw_deg(root_aa: np.ndarray) -> np.ndarray:
    """Heading of a root orientation (T, 3) axis-angle in a y-up world: the yaw of the pelvis forward axis (SMPL
    faces +z in its rest pose) projected on the floor, degrees in (-180, 180]; NaN rows stay NaN."""
    root_aa = np.asarray(root_aa, float); ok = np.isfinite(root_aa).all(1); out = np.full(len(root_aa), np.nan)
    if ok.any():
        f = R.from_rotvec(root_aa[ok]).apply(np.array([0.0, 0.0, 1.0]))
        out[ok] = np.degrees(np.arctan2(f[:, 0], f[:, 2]))
    return out


def rate_in_runs(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d/dt of x, (T,) or (T, D), on the clock t (s), inside each run of finite rows of two or more; NaN elsewhere."""
    out = np.full(x.shape, np.nan); ok = np.isfinite(x).all(1) if x.ndim == 2 else np.isfinite(x)
    e = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(e[::2], e[1::2]):
        if b - a >= 2:
            out[a:b] = np.gradient(x[a:b], t[a:b], axis=0)
    return out


def heading_frame(root_aa: np.ndarray):
    """A root track's heading and the root with the heading taken out: (yaw, yaw_unwrapped, root_heading_rel).
    yaw is heading_yaw_deg; yaw_unwrapped is the same angle made continuous inside each finite run;
    root_heading_rel is the root turned by -yaw about the vertical, so the body faces +z whichever way it walks."""
    root_aa = np.asarray(root_aa, float); T = len(root_aa)
    yaw = heading_yaw_deg(root_aa); yaw_u = np.full(T, np.nan); ok = np.isfinite(yaw)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        yaw_u[a:b] = np.degrees(np.unwrap(np.radians(yaw[a:b])))
    root_hr = np.full((T, 3), np.nan)
    with np.errstate(all="ignore"):
        root_hr[ok] = (R.from_euler("y", -yaw[ok], degrees=True) * R.from_rotvec(root_aa[ok])).as_rotvec()
    return yaw, yaw_u, root_hr


def heading_rel(vec: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    """World vectors (T, 3) turned by -yaw about the vertical: x to the body's left, y up, z forward. NaN where
    the vector or the heading is missing."""
    out = np.full((len(vec), 3), np.nan); ok = np.isfinite(vec).all(1) & np.isfinite(yaw_deg)
    with np.errstate(all="ignore"):
        out[ok] = R.from_euler("y", -yaw_deg[ok], degrees=True).apply(vec[ok])
    return out


def circular_median_deg(d: np.ndarray) -> float:
    """Median of angles (deg) that is insensitive to the +-180 wrap: centre on the circular mean, take the median
    of the wrapped deviations."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    c = np.degrees(np.arctan2(np.sin(np.radians(d)).mean(), np.cos(np.radians(d)).mean()))
    dev = (d - c + 180.0) % 360.0 - 180.0
    return float((c + np.median(dev) + 180.0) % 360.0 - 180.0)


def fit_yaw(aa_est: np.ndarray, aa_gt: np.ndarray, mask: np.ndarray) -> float:
    """Yaw (deg, about world y) that best maps the estimated root orientations onto the truth (analysis only)."""
    best = (1e9, 0.0)
    for th in np.arange(-180, 180, 1.0):
        Ry = R.from_euler("y", th, degrees=True)
        e = np.degrees((Ry * R.from_rotvec(aa_est[mask])).inv().__mul__(R.from_rotvec(aa_gt[mask])).magnitude()).mean()
        if e < best[0]:
            best = (e, th)
    return best[1]
