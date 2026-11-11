"""The lifter never runs across an exit: an absence of the subject longer than absence_split_s ends the chunk and a new
one starts at the re-entry; shorter gaps are bridged (and flagged visible=False downstream)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.config import DEFAULT_CONFIG          # noqa: E402
from wgvp.session import plan_chunks            # noqa: E402
from wgvp.tracking import TrackSet              # noqa: E402


def _scene(absences, fps=30.0, dur_s=50.0):
    """A one-view TrackSet of dur_s seconds in which the subject is present except during the given [t0, t1) spans."""
    F = int(dur_s * fps); t = np.arange(F) / fps
    present = np.ones(F, bool)
    for a, b in absences:
        present[(t >= a) & (t < b)] = False
    pos = np.flatnonzero(present)
    ts = TrackSet(video="v", view_id="full", box=[0, 0, 640, 480], width=640, height=480, fps_nominal=fps, frame_idx=np.arange(F), pts_ms=t * 1000.0,
                  det_frame=pos, det_tid=np.ones(len(pos), int), det_box=np.tile([100.0, 50.0, 200.0, 400.0], (len(pos), 1)), det_conf=np.ones(len(pos)),
                  det_kp=np.zeros((len(pos), 17, 3)), det_app=np.zeros((len(pos), 48)), det_vis=np.full(len(pos), 2, np.int8), grid=np.zeros((F, 18, 32), np.uint8))
    subj = {"pos": pos, "pts_ms": ts.pts_ms[pos], "box": ts.det_box}
    imu = {"duration_s": dur_s - 4.0, "events": [("Standing", 0.0, 3.0), ("Walk", 3.0, dur_s - 4.0)]}
    return ts, subj, imu


def _spans(ts, chunks):
    return [(round(ts.pts_ms[c["pos"][0]] / 1000, 2), round(ts.pts_ms[c["pos"][-1]] / 1000, 2)) for c in chunks]


def test_long_absence_splits_and_short_gap_bridges():
    prm = dict(DEFAULT_CONFIG["params"])
    ts, subj, imu = _scene([(20.0, 23.3), (35.0, 36.0)])       # a 3.3-s exit and a 1-s loss, IMU t=0 at video 2 s
    chunks = plan_chunks(ts, subj, 2.0, imu, prm, 30.0)
    assert len(chunks) == 2, _spans(ts, chunks)
    (a0, a1), (b0, b1) = _spans(ts, chunks)
    assert a1 <= 20.0 + 1.05 and b0 >= 23.3 - 1.05            # at most the 1-s margin into the absence on either side
    frames_in_exit = sum(int(((ts.pts_ms[c["pos"]] / 1000 >= 21.1) & (ts.pts_ms[c["pos"]] / 1000 < 22.3)).sum()) for c in chunks)
    assert frames_in_exit == 0                                 # the middle of the exit is in no chunk at all
    assert b1 > 36.0                                           # the 1-s loss did not split the second chunk


def test_no_absence_one_chunk_and_threshold_is_configurable():
    prm = dict(DEFAULT_CONFIG["params"])
    ts, subj, imu = _scene([])
    assert len(plan_chunks(ts, subj, 2.0, imu, prm, 30.0)) == 1
    ts, subj, imu = _scene([(20.0, 21.5)])
    assert len(plan_chunks(ts, subj, 2.0, imu, prm, 30.0)) == 1                     # 1.5 s < 2 s: bridged
    assert len(plan_chunks(ts, subj, 2.0, imu, {**prm, "absence_split_s": 1.0}, 30.0)) == 2   # unless the threshold says otherwise


def _rest_scene(absences, boxes=None):
    """_scene with the subject standing still for the first 13 s on the IMU clock (video 2-15 s), then walking."""
    ts, subj, imu = _scene(absences)
    imu["events"] = [("Standing", 0.0, 13.0), ("Walk", 13.0, imu["duration_s"])]
    if boxes is not None:
        subj["box"] = boxes(ts.pts_ms[subj["pos"]] / 1000.0)
    return ts, subj, imu


def test_rest_hold_keeps_the_box_where_the_imu_says_the_body_is_at_rest():
    """A detector that loses a standing subject for 4 s does not end the chunk: the IMU labels the body at rest, the
    boxes on either side of the loss are the same box, so the frames keep it and are flagged rest_held. The same loss
    while walking is an absence, and so is one whose two ends do not show the same box."""
    from wgvp.session import rest_hold, subject_presence
    prm = dict(DEFAULT_CONFIG["params"])
    ts, subj, imu = _rest_scene([(6.0, 10.0)])                                         # lost at video 6-10 s = IMU 4-8 s: standing
    chunks = plan_chunks(ts, subj, 2.0, imu, prm, 30.0); assert len(chunks) == 1, _spans(ts, chunks)
    present, box, held = subject_presence(ts, subj, 2.0, imu, prm)
    t = ts.pts_ms / 1000.0; gap = (t >= 6.0) & (t < 10.0)
    assert held[gap].all() and not held[~gap].any() and not present[gap].any() and np.allclose(box[gap], [100.0, 50.0, 200.0, 400.0])
    assert len(plan_chunks(ts, subj, 2.0, imu, {**prm, "rest_hold": False}, 30.0)) == 2                       # the switch
    ts, subj, imu = _rest_scene([(20.0, 24.0)])                                        # the same loss while walking: an exit
    assert len(plan_chunks(ts, subj, 2.0, imu, prm, 30.0)) == 2 and not subject_presence(ts, subj, 2.0, imu, prm)[2].any()
    moved = lambda tt: np.where((tt < 8.0)[:, None], [100.0, 50.0, 200.0, 400.0], [300.0, 50.0, 400.0, 400.0])
    ts, subj, imu = _rest_scene([(6.0, 10.0)], boxes=moved)                            # labelled at rest, but the box moved: no hold
    assert len(plan_chunks(ts, subj, 2.0, imu, prm, 30.0)) == 2 and not subject_presence(ts, subj, 2.0, imu, prm)[2].any()
    # one-sided: no detection until 5 s into the stand, and none in its last 3 s. The stand's frames take the first
    # and the last detection's box, up to 0.3 s short of the walk label (the recording's own start is not trimmed).
    ts, subj, imu = _rest_scene([(0.0, 7.0), (12.0, 16.5)])
    present, box, held = subject_presence(ts, subj, 2.0, imu, prm); ti = t - 2.0
    assert held[(ti >= 0.0) & (t < 7.0)].all() and not held[ti < -1e-9].any()
    assert held[(t >= 12.0) & (ti <= 12.7 - 1e-6)].all() and not held[ti > 12.7 + 1e-6].any()
    chunks = plan_chunks(ts, subj, 2.0, imu, prm, 30.0); assert len(chunks) == 1 and abs(_spans(ts, chunks)[0][0] - 2.0) < 0.05   # lifted from the IMU start
    h, bh = rest_hold(ti, present, box, {"duration_s": 46.0, "events": [("Walk", 0.0, 46.0)]}); assert not h.any() and np.isnan(bh).all()   # no rest label, no hold
