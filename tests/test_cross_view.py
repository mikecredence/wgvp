"""Cross-view confirmation of the subject: a side view where the walker is small (no usable pelvis bob) takes its
subject from the along view's confirmed one by their progress along the walk at zero lag. Fabricated two-view
tracking of a back-and-forth walk: the along view carries the subject (body width in pixels follows the distance),
the side view holds the subject (hip column follows the same distance), a static operator, a bystander swaying in
place and a co-walker 0.8 s behind, who must be flagged, not chosen."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.selection import cross_view_selection, progress_signals   # noqa: E402
from wgvp.tracking import TrackSet                                 # noqa: E402

FPS = 30.0


def _kp(x, y, width, conf=0.9):
    """COCO-17 keypoints of an upright person at hip column x, hip row y, with shoulder/hip/knee widths scaled by `width`."""
    k = np.zeros((len(x), 17, 3)); k[:, :, 2] = conf
    for a, b, w, dy in ((5, 6, 1.2, -0.6), (11, 12, 1.0, 0.0), (13, 14, 0.7, 0.5)):
        k[:, a, 0] = x - 0.5 * w * width; k[:, b, 0] = x + 0.5 * w * width; k[:, a, 1] = k[:, b, 1] = y + dy * width * 2
    return k


def _trackset(view, box, people):
    """people: list of (tid, frames, kp) -> one TrackSet with those detections."""
    F = 1500; pos, tid, kp = [], [], []
    for t, frames, k in people:
        pos.append(frames); tid.append(np.full(len(frames), t)); kp.append(k)
    pos = np.concatenate(pos); tid = np.concatenate(tid); kp = np.concatenate(kp)
    box_xyxy = np.c_[kp[:, 11, 0] - 30, kp[:, 11, 1] - 120, kp[:, 12, 0] + 30, kp[:, 12, 1] + 150]
    return TrackSet(video="v", view_id=view, box=box, width=1420, height=600, fps_nominal=FPS, frame_idx=np.arange(F), pts_ms=np.arange(F) / FPS * 1000.0,
                    det_frame=pos, det_tid=tid, det_box=box_xyxy, det_conf=np.ones(len(pos)), det_kp=kp, det_app=np.zeros((len(pos), 48)),
                    det_vis=np.full(len(pos), 2, np.int8), grid=np.zeros((F, 18, 32), np.uint8))


def _scene(seed=0):
    rng = np.random.default_rng(seed); F = 1500; t = np.arange(F) / FPS
    d = 2.5 + 2.0 * np.sin(2 * np.pi * t / 20.0)                       # distance from the along camera, m: a back-and-forth walk, 20-s cycle
    # along view: the subject's pixel width is inversely proportional to the distance; the hips stay near the centre
    along_frames = np.arange(F); w_along = 90.0 / d + rng.normal(0, 1.0, F)
    along = _trackset("left", [0, 0, 520, 600], [(1, along_frames, _kp(np.full(F, 260.0) + rng.normal(0, 2, F), 300 + 20 * (2.5 - d), w_along))])
    # side view: progress along the walk is left-right; the walker is small (bob invisible in the noise)
    x_subj = 700 + 80 * d + rng.normal(0, 1.5, F); subj = _kp(x_subj, np.full(F, 150.0) + rng.normal(0, 1.5, F), np.full(F, 20.0))
    x_co = 700 + 80 * np.interp(t - 0.8, t, d) + rng.normal(0, 1.5, F); cow = _kp(x_co, np.full(F, 165.0), np.full(F, 20.0))          # a co-walker 0.8 s behind
    oper = _kp(np.full(F, 1200.0) + rng.normal(0, 1.0, F), np.full(F, 400.0), np.full(F, 40.0))                                         # the seated operator
    byst = _kp(950 + 6 * np.sin(2 * np.pi * t / 3.0) + rng.normal(0, 1.0, F), np.full(F, 120.0), np.full(F, 18.0))                     # a bystander swaying in place
    side = _trackset("right", [520, 0, 1420, 375], [(3, along_frames, subj), (4, along_frames, cow), (5, along_frames, oper), (6, along_frames, byst)])
    return along, side


def test_progress_signals_follow_the_walk():
    along, side = _scene(); n = 1500
    ps = progress_signals(along.track(1), n, FPS); xs = progress_signals(side.track(3), n, FPS)
    ok = np.isfinite(ps["scale"]) & np.isfinite(xs["x"]); r = np.corrcoef(ps["scale"][ok], xs["x"][ok])[0, 1]
    assert r < -0.9, r                                                    # width grows as the walker nears the along camera, x grows as they walk away (log vs linear: not a straight line)


def test_cross_view_takes_the_subject_and_flags_the_co_walker():
    along, side = _scene()
    xv = cross_view_selection(side, along.track(1), FPS)
    assert xv["ok"] and xv["track_ids"] == [3], xv
    assert abs(xv["candidates"][3]["r"]) > 0.95 and abs(xv["candidates"][5]["r"]) < 0.5 and abs(xv["candidates"][6]["r"]) < 0.5
    assert abs(xv["candidates"][4]["r"]) > 0.9 and xv["margin"] < 0.1 and "moves with the subject" in xv["reason"]    # the co-walker correlates nearly as well: flagged, not chosen


def test_cross_view_without_the_co_walker_is_clean():
    along, side = _scene()
    keep = side.det_tid != 4
    side2 = TrackSet(**{**side.__dict__, "det_frame": side.det_frame[keep], "det_tid": side.det_tid[keep], "det_box": side.det_box[keep], "det_conf": side.det_conf[keep],
                         "det_kp": side.det_kp[keep], "det_app": side.det_app[keep], "det_vis": side.det_vis[keep]})
    xv = cross_view_selection(side2, along.track(1), FPS)
    assert xv["ok"] and xv["track_ids"] == [3] and xv["margin"] > 0.4 and xv["reason"] == ""


def test_refine_lag_stays_on_the_lights_peak():
    """A lag taken from the sync light is refined only within a tenth of a second: the pelvis bob repeats every
    step, and the wider window used for the match's own lag slides onto the neighbouring step's stronger peak,
    half a step off the light (0.54 s on HC901)."""
    from wgvp.selection import refine_lag
    from wgvp.signals import FS_MATCH
    axis = np.arange(-5.0, 5.0, 1.0 / FS_MATCH)
    def bump(c, h): return h * np.exp(-0.5 * ((axis - c) / 0.08) ** 2)
    R = bump(3.0, 0.35) + bump(3.55, 0.50) + bump(2.45, 0.45)                     # the light's peak and the two neighbouring steps
    curves = [(axis, R, 30.0)]                                                  # (lags, r on those lags, weight)
    wide = refine_lag(curves, 3.0, 1.0); narrow = refine_lag(curves, 3.0, 0.1)
    assert abs(wide - 3.55) < 0.02, wide                                         # what the old refinement did
    assert abs(narrow - 3.0) < 0.02, narrow
    assert refine_lag(curves, 3.0, 0.0) == 3.0 or abs(refine_lag(curves, 3.0, 0.0) - 3.0) < 0.02


def _standing_scene(seed=0, split_at=None, operator_x=1250.0, operator_y=400.0, overlap=0):
    """The walker stands at the far end for the first 12 s, then walks the 20-s back-and-forth; the side view's subject
    is one fragment, or split at `split_at` frames into a standing fragment and a walking one (a tracker that lost the
    id when the walk began). The operator's track ends at the same frame, far from the junction."""
    rng = np.random.default_rng(seed); F = 1500; t = np.arange(F) / FPS
    d = np.where(t < 12.0, 4.5, 2.5 + 2.0 * np.cos(2 * np.pi * (t - 12.0) / 20.0))
    frames = np.arange(F); w_along = 90.0 / d + rng.normal(0, 1.0, F)
    along = _trackset("left", [0, 0, 520, 600], [(1, frames, _kp(np.full(F, 260.0) + rng.normal(0, 2, F), 300 + 20 * (2.5 - d), w_along))])
    subj = _kp(700 + 80 * d + rng.normal(0, 1.5, F), np.full(F, 150.0) + rng.normal(0, 1.5, F), np.full(F, 20.0))
    oper = _kp(np.full(F, operator_x) + rng.normal(0, 1.0, F), np.full(F, operator_y), np.full(F, 40.0))
    if split_at is None:
        people = [(3, frames, subj), (5, frames[:split_at] if split_at else frames, oper)]
    else:
        people = [(7, frames[:split_at + overlap], subj[:split_at + overlap]), (3, frames[split_at:], subj[split_at:]), (5, frames[:split_at], oper[:split_at])]
    return along, _trackset("right", [520, 0, 1420, 375], people)


def test_cross_view_keeps_the_standing_frames_of_an_attached_fragment():
    """Twelve seconds of standing inside the subject's fragment: the window test must not read the correlation of two
    standing people's keypoint noise as an identity switch and drop them."""
    along, side = _standing_scene()
    xv = cross_view_selection(side, along.track(1), FPS)
    assert xv["ok"] and xv["track_ids"] == [3], xv
    assert xv["candidates"][3].get("frames_dropped", 0) == 0, xv["candidates"][3]
    assert 3 not in xv["frame_masks"]


def test_cross_view_attaches_a_standing_fragment_by_continuity():
    """The tracker split the subject when the walk began: the standing fragment has no progress to correlate, and is
    attached because it ends where the walking fragment begins; the operator's track, ending at the same frame but
    elsewhere in the panel, is not."""
    along, side = _standing_scene(split_at=360)
    xv = cross_view_selection(side, along.track(1), FPS)
    assert xv["ok"] and xv["track_ids"] == [3, 7] and xv["attached_by_continuity"] == [7], xv
    assert 7 not in xv["frame_masks"] and 3 not in xv["frame_masks"]
    along2, side2 = _standing_scene(split_at=360, operator_x=1075.0, operator_y=150.0)   # two fragments fit the junction: attach neither
    xv2 = cross_view_selection(side2, along2.track(1), FPS)
    assert xv2["track_ids"] == [3] and xv2["attached_by_continuity"] == []
    along3, side3 = _standing_scene(split_at=360, overlap=2)                         # the tracker handed the id over with two frames in common
    xv3 = cross_view_selection(side3, along3.track(1), FPS)
    assert xv3["track_ids"] == [3, 7] and xv3["attached_by_continuity"] == [7], xv3


def test_window_test_reaches_the_end_of_a_fragment():
    """A tracker id change at frame 291 leaves a 291-frame fragment: on the hop grid alone its frames 150-224 sit in
    one window (75-224), and a fault inside that window but before them (the box off the walker over frames
    100-140) dropped them with it. A last window flush with the fragment's end (141-290) judges them a second time
    and keeps them; the faulty frames still go."""
    rng = np.random.default_rng(0); F = 1500; t = np.arange(F) / FPS; cut = 291
    d = 2.5 + 2.0 * np.sin(2 * np.pi * t / 10.0); frames = np.arange(F); w_along = 90.0 / d + rng.normal(0, 1.0, F)
    along = _trackset("left", [0, 0, 520, 600], [(1, frames, _kp(np.full(F, 260.0) + rng.normal(0, 2, F), 300 + 20 * (2.5 - d), w_along))])
    x = 700 + 80 * d + rng.normal(0, 1.5, F); x[100:141] = x[141] - 3.0 * (x[100:141] - x[141])    # the box runs the wrong way for 1.4 s
    subj = _kp(x, np.full(F, 150.0) + rng.normal(0, 1.5, F), np.full(F, 20.0))
    side = _trackset("right", [520, 0, 1420, 375], [(7, frames[:cut], subj[:cut]), (3, frames[cut:], subj[cut:])])
    xv = cross_view_selection(side, along.track(1), FPS)
    assert xv["ok"] and xv["track_ids"] == [3, 7], xv
    keep = xv["frame_masks"][7]
    assert not keep[100:141].any(), "the faulty frames must go"
    assert keep[141:].all(), f"frames after the fault were dropped: {np.where(~keep)[0].max()}"
    assert 3 not in xv["frame_masks"]
