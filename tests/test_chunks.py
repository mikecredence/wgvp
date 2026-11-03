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
