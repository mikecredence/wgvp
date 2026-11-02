"""Optional second sync anchor: Walk intervals labelled ON THE VIDEO (MATLAB Video Labeler scene labels) versus the
recording CSV's GeneralEvent Walk column (same labels on the IMU timeline). Accepts a CSV export with either
(a) a time column + a 'Walk' column (0/1, true/false; the MATLAB timetable export of scene labels), or
(b) interval rows with a label column and start/end time columns. MATLAB v5 .mat files are read with scipy; v7.3
session files need h5py (not installed by default) - the README gives a two-line MATLAB export to CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .signals import masked_xcorr, parabolic_peak

ENV_FS = 20.0


def _truthy(v) -> bool:
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "walk") or (s.replace(".", "", 1).isdigit() and float(s) > 0)


def _to_seconds(v) -> float | None:
    s = str(v).strip()
    try:
        return float(s)
    except ValueError:
        pass
    for fmt in ("sec", "s"):
        if s.endswith(fmt):
            try:
                return float(s[:-len(fmt)])
            except ValueError:
                pass
    if ":" in s:   # hh:mm:ss.fff or mm:ss.fff
        parts = s.split(":")
        try:
            return sum(float(x) * 60 ** i for i, x in enumerate(reversed(parts)))
        except ValueError:
            return None
    return None


def read_walk_intervals(path: Path) -> list[tuple[float, float]]:
    p = Path(path)
    if p.suffix.lower() == ".mat":
        return _read_mat(p)
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    cols = {c.lower().strip(): c for c in rows[0].keys() if c}
    tcol = next((cols[c] for c in cols if c in ("time", "t", "timestamp", "video_time", "time_s")), None)
    wcol = next((cols[c] for c in cols if c == "walk"), None)
    if tcol and wcol:                                            # timetable: Time, Standing, Walk, Turn, ...
        t = np.array([_to_seconds(r[tcol]) for r in rows], dtype=np.float64); w = np.array([_truthy(r[wcol]) for r in rows])
        ok = np.isfinite(t); t, w = t[ok], w[ok]
        iv = []; start = None
        for i in range(len(t)):
            if w[i] and start is None:
                start = t[i]
            if (not w[i] or i == len(t) - 1) and start is not None:
                iv.append((float(start), float(t[i]))); start = None
        return iv
    lcol = next((cols[c] for c in cols if c in ("label", "name", "event", "scene_label")), None)
    s0 = next((cols[c] for c in cols if c in ("start", "start_time", "t0", "starttime", "start_s")), None)
    s1 = next((cols[c] for c in cols if c in ("end", "end_time", "t1", "endtime", "end_s")), None)
    if lcol and s0 and s1:
        return [(_to_seconds(r[s0]), _to_seconds(r[s1])) for r in rows if str(r[lcol]).strip().lower() == "walk" and _to_seconds(r[s0]) is not None and _to_seconds(r[s1]) is not None]
    return []


def _read_mat(p: Path) -> list[tuple[float, float]]:
    try:
        from scipy.io import loadmat
        d = loadmat(p, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        raise RuntimeError(f"{p.name} is a MATLAB v7.3 file; export the labels to CSV in MATLAB (see README) or install h5py")
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if hasattr(v, "dtype") and v.dtype.names and "Walk" in v.dtype.names and "Time" in v.dtype.names:
            t = np.asarray(v["Time"], float).ravel(); w = np.asarray(v["Walk"]).ravel().astype(bool)
            iv, start = [], None
            for i in range(len(t)):
                if w[i] and start is None:
                    start = t[i]
                if (not w[i] or i == len(t) - 1) and start is not None:
                    iv.append((float(start), float(t[i]))); start = None
            return iv
    raise RuntimeError(f"{p.name}: no Time/Walk table found; export the scene labels to CSV (see README)")


def walk_label_offset(video_walk: list[tuple[float, float]], imu_events: list, video_len_s: float, imu_len_s: float, fs: float = ENV_FS) -> dict:
    """Cross-correlate the video-time Walk indicator with the IMU-time Walk indicator -> offset (video time of IMU t=0)."""
    if not video_walk:
        return {"available": False, "reason": "no Walk intervals in the label export"}
    tv = np.arange(0, video_len_s, 1 / fs); ti = np.arange(0, imu_len_s, 1 / fs)
    v = np.zeros(len(tv)); u = np.zeros(len(ti))
    for a, b in video_walk:
        v[(tv >= a) & (tv <= b)] = 1
    for lab, a, b in imu_events:
        if lab == "Walk":
            u[(ti >= a) & (ti <= b)] = 1
    if u.sum() == 0:
        return {"available": False, "reason": "no Walk events in the recording CSV"}
    lags, r, n = masked_xcorr(v - v.mean(), u - u.mean(), len(v) + len(u), min_overlap=int(0.5 * len(u)))
    if not np.isfinite(r).any():
        return {"available": False, "reason": "no overlap"}
    i = int(np.nanargmax(r))
    return {"available": True, "offset_s": float((lags[0] + parabolic_peak(r, i)) / fs), "r": float(r[i]), "n_video_walk_bouts": len(video_walk),
            "video_walk_s": float(sum(b - a for a, b in video_walk)), "imu_walk_s": float(u.sum() / fs)}
