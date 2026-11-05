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
