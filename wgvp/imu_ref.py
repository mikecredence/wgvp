"""IMU-side reference signals for one WearGait recording, read DIRECTLY from the recording CSV that sits beside
the videos (pelvis accelerometer, GeneralEvent annotations, walkway foot contacts). Only two to six columns of the
297-column file are read, with the stdlib csv module (pandas is used when available, it is faster). The derived
signals are cached as a small NPZ in the work folder. Content per recording:

  fs                100 (Hz)
  t                 (N,) seconds from the IMU recording start
  acc_up            (N,) pelvis (LowerBack) free vertical acceleration, gait band-passed, m/s^2 (world up)
  disp_up           (N,) the same signal integrated twice inside the gait band (arbitrary units, in phase with
                         the pelvis vertical displacement) - this is what the video pelvis is matched against
  env               (N,) RMS envelope (1 s) of acc_up - the activity pattern used for the coarse match
  event_label       (K,) str   GeneralEvent segments (Standing / Walk / Turn / unlabeled)
  event_t0, event_t1 (K,) s
  duration_s, n_rows, nan_fraction, source_sha256 (provenance of the CSV, no content), recording_id, subject
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .signals import bandpass, double_integrate, rms_envelope, GAIT_BAND

IMU_FS = 100.0
PELVIS_SENSOR = "LowerBack"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_time(col) -> np.ndarray:
    """WearGait 'Time' is '0.01 sec' strings; accept plain floats too."""
    out = np.empty(len(col), dtype=np.float64)
    for i, v in enumerate(col):
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[i] = float(v)
        else:
            out[i] = float(str(v).split()[0])
    return out


CONTACT_COLS = ["L Foot Contact", "R Foot Contact"]


def _read_columns(csv_path: Path, wanted: list[str]) -> dict:
    """Read the wanted columns (those present) as object/float arrays; stdlib csv, pandas if importable."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, usecols=lambda c: c in wanted, low_memory=False)
        return {c: df[c].to_numpy() for c in df.columns}
    except ImportError:
        pass
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rd = _csv.reader(fh); header = next(rd)
        idx = {c: header.index(c) for c in wanted if c in header}
        cols = {c: [] for c in idx}
        for row in rd:
            if not row:
                continue
            for c, i in idx.items():
                cols[c].append(row[i] if i < len(row) else "")
    out = {}
    for c, vals in cols.items():
        if c in ("Time", "GeneralEvent", "ClinicalEvent"):
            out[c] = np.array(vals, dtype=object)
        else:
            out[c] = np.array([float(v) if v not in ("", "nan", "NaN") else np.nan for v in vals], dtype=np.float64)
    return out


def make_imu_ref(csv_path: str | Path, out_path: str | Path, recording_id: str | None = None,
                 subject: str | None = None, fs: float = IMU_FS) -> dict:
    csv_path = Path(csv_path); out_path = Path(out_path)
    cols = ["Time", "GeneralEvent", f"{PELVIS_SENSOR}_FreeAcc_U"] + CONTACT_COLS
    data = _read_columns(csv_path, cols)
    if f"{PELVIS_SENSOR}_FreeAcc_U" not in data:
        raise ValueError(f"{csv_path.name}: no {PELVIS_SENSOR}_FreeAcc_U column (is this a WearGait recording export?)")
    n_rows_file = len(data[f"{PELVIS_SENSOR}_FreeAcc_U"])
    t = _parse_time(data["Time"]) if "Time" in data else np.arange(n_rows_file) / fs
    acc = np.asarray(data[f"{PELVIS_SENSOR}_FreeAcc_U"], dtype=np.float64)
    nan_fraction = float(np.mean(~np.isfinite(acc)))
    dt = np.median(np.diff(t)) if len(t) > 1 else 1 / fs
    if abs(dt * fs - 1) > 0.02:
        raise ValueError(f"{csv_path.name}: sample interval {dt:.4f}s is not {1/fs:.4f}s")
    # uniform grid in case of gaps/jitter
    n = int(round(t[-1] * fs)) + 1
    grid = np.arange(n) / fs
    ok = np.isfinite(acc)
    acc_u = np.interp(grid, t[ok], acc[ok]) if ok.sum() > 10 else np.full(n, np.nan)
    acc_bp = bandpass(acc_u, fs, *GAIT_BAND)
    disp = double_integrate(acc_u, fs, *GAIT_BAND)
    env = rms_envelope(acc_bp, fs)
    labels, t0s, t1s = [], [], []
    contact = np.full(n, np.nan)
    if all(c in data for c in CONTACT_COLS):
        lc = np.asarray(data[CONTACT_COLS[0]], dtype=np.float64); rc = np.asarray(data[CONTACT_COLS[1]], dtype=np.float64)
        anyc = np.where(np.isfinite(lc) | np.isfinite(rc), np.nan_to_num(lc) + np.nan_to_num(rc), np.nan)
        okc = np.isfinite(anyc)
        if okc.sum() > 10:
            contact = np.interp(grid, t[okc], anyc[okc], left=np.nan, right=np.nan)
    if "GeneralEvent" in data:
        g = np.array([str(v) for v in data["GeneralEvent"]])
        start = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or g[i] != g[start]:
                labels.append(g[start]); t0s.append(float(t[start])); t1s.append(float(t[i - 1]) + 1 / fs); start = i
    rec = recording_id or csv_path.stem
    subj = subject or csv_path.stem.split("_")[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fs=fs, t=grid.astype(np.float32), acc_up=acc_bp.astype(np.float32),
                        disp_up=disp.astype(np.float32), env=env.astype(np.float32),
                        event_label=np.array(labels), event_t0=np.array(t0s, np.float32), event_t1=np.array(t1s, np.float32),
                        contact=contact.astype(np.float32), contact_fraction=float(np.nanmean(contact > 0)) if np.isfinite(contact).any() else -1.0,
                        duration_s=float(grid[-1]), n_rows=int(n_rows_file), nan_fraction=nan_fraction,
                        source_sha256=_sha256(csv_path), source_name=csv_path.name, recording_id=rec, subject=subj,
                        schema="wgvp.imu_ref.v1")
    walk_s = float(sum(b - a for l, a, b in zip(labels, t0s, t1s) if l == "Walk"))
    return {"recording_id": rec, "subject": subj, "duration_s": float(grid[-1]), "walk_s": walk_s,
            "nan_fraction": nan_fraction, "out": str(out_path)}


def load_imu_ref(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    for k in ("fs", "duration_s", "nan_fraction", "contact_fraction"):
        if k in d:
            d[k] = float(d[k])
    for k in ("recording_id", "subject", "source_sha256", "source_name", "schema"):
        d[k] = str(d[k])
    d["events"] = [(str(l), float(a), float(b)) for l, a, b in zip(d["event_label"], d["event_t0"], d["event_t1"])]
    return d


def summarize_imu_ref(d: dict) -> dict:
    ev = d["events"]
    return {"recording_id": d["recording_id"], "subject": d["subject"], "duration_s": d["duration_s"],
            "walk_s": float(sum(b - a for l, a, b in ev if l == "Walk")),
            "turn_s": float(sum(b - a for l, a, b in ev if l == "Turn")),
            "standing_s": float(sum(b - a for l, a, b in ev if l == "Standing")),
            "n_walk_bouts": int(sum(1 for l, a, b in ev if l == "Walk")),
            "walkway_contact_fraction": d.get("contact_fraction", -1.0),
            "nan_fraction": d["nan_fraction"], "source_sha256": d["source_sha256"], "source_name": d.get("source_name")}


def load_recording(csv_path: Path, cache_dir: Path, recording_id: str, subject: str) -> dict:
    """Derived reference for a recording CSV, cached in cache_dir (re-made when the CSV changes)."""
    from .video_io import sha256_file
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{recording_id}_imu_ref.npz"
    if out.exists():
        d = load_imu_ref(out)
        if d["source_sha256"] == sha256_file(csv_path):
            return d
    make_imu_ref(csv_path, out, recording_id=recording_id, subject=subject)
    return load_imu_ref(out)


def simulate_imu_ref_from_video_signal(t_video: np.ndarray, hip_up: np.ndarray, offset_s: float, fs: float = IMU_FS,
                                       duration_s: float | None = None, noise_sd: float = 0.3, seed: int = 0) -> dict:
    """TEST HELPER: fabricate an IMU reference from a track's own pelvis vertical signal (video PTS time), placed so
    that IMU t=0 sits at video time ``offset_s``. The 'acceleration' is the second derivative of the smoothed
    displacement, with white noise added; disp/env are derived from it exactly as make_imu_ref would."""
    rng = np.random.default_rng(seed)
    t_imu = np.asarray(t_video) - offset_s
    if duration_s is None:
        duration_s = float(np.nanmax(t_imu))
    n = int(round(duration_s * fs)) + 1
    grid = np.arange(n) / fs
    ok = np.isfinite(hip_up) & np.isfinite(t_imu)
    disp_true = np.interp(grid, t_imu[ok], np.asarray(hip_up)[ok], left=np.nan, right=np.nan)
    disp_true = bandpass(disp_true, fs)
    disp_f = np.where(np.isfinite(disp_true), disp_true, 0.0)
    acc = np.gradient(np.gradient(disp_f, 1 / fs), 1 / fs)
    acc = acc / (np.nanstd(acc) + 1e-9) + noise_sd * rng.standard_normal(n)
    acc_bp = bandpass(acc, fs); disp = double_integrate(acc, fs); env = rms_envelope(acc_bp, fs)
    return {"fs": fs, "t": grid, "acc_up": acc_bp, "disp_up": disp, "env": env, "events": [],
            "duration_s": float(grid[-1]), "recording_id": "simulated", "subject": "simulated", "nan_fraction": 0.0,
            "source_sha256": "", "source_name": "", "schema": "wgvp.imu_ref.v1(simulated)"}
