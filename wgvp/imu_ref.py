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
  walkway_*         the pressure walkway, when the export carries it (see parse_walkway): per sample the centre of
                         each labelled footprint on the mat (metres), and the stances those samples form
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
SCHEMA = "wgvp.imu_ref.v2"      # v2: walkway cells parsed into footprint centres and stances


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


def _safe_float(col) -> np.ndarray:
    """Numbers where the cell is a number, NaN for anything else (empty, text, 'a|b|c' lists)."""
    out = np.full(len(col), np.nan)
    for i, v in enumerate(col):
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[i] = float(v); continue
        try:
            out[i] = float(str(v).strip())
        except ValueError:
            pass
    return out


CONTACT_COLS = ["L Foot Contact", "R Foot Contact"]
WALKWAY_COLS = ["Walkway_X", "Walkway_Y"]                                  # the pressure walkway's loaded cells, per sample (see parse_walkway)
WALKWAY_EXTRA = ["WalkwayFoot", "WalkwayPressureLevel"]                    # their foot labels and pressure levels, same lists
WALKWAY_CELL_CM = 1.27          # the sensel pitch of the GAITRite / Zeno family (half an inch): a 384 x 48 grid is 4.88 x 0.61 m
WALKWAY_MIN_CELLS = 5           # cells under one label before the sample counts as a footprint
STANCE_MIN_S = 0.15             # a footprint present for less than this is not a stance
STANCE_GAP_S = 0.12             # samples of one footprint may be this far apart (the export drops rows)
STANCE_DRIFT_M = 0.25           # the centre of one footprint moves heel to toe by less than this; more is the next step
RAW_COLS = ("Time", "GeneralEvent", "ClinicalEvent", *WALKWAY_COLS, *WALKWAY_EXTRA)


def _split(v) -> list[str]:
    """One walkway cell -> its '|'-separated tokens ([] for an empty / NaN cell)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return []
    s = str(v).strip()
    return [] if (s == "" or s.lower() == "nan") else s.split("|")


def parse_walkway(t: np.ndarray, wx, wy, wfoot=None, wpress=None, cell_m: float = WALKWAY_CELL_CM / 100.0) -> dict:
    """The pressure walkway as the WearGait export carries it: at every IMU sample (10 ms) the lists of loaded
    cells, 'x1|x2|...' in Walkway_X and the matching 'y1|y2|...' in Walkway_Y (integer cell indices along and
    across the mat, 0-383 by 0-47 on the 4.88 x 0.61 m GAITRite), with the footprint label of every cell in
    WalkwayFoot ('L', 'R', or 'Other 1' for anything the software did not assign) and its pressure level in
    WalkwayPressureLevel. Empty cells mean nobody is on the mat; the mat records all through the session, whatever
    the task. Per sample and per labelled foot this returns the pressure-weighted centre of the footprint in
    metres, and groups the consecutive samples of one footprint into stances (first/last sample, median centre).
    A legacy form, one plain number per row (a footfall position in cm), is accepted too (`format` 'footfall_cm').
    Returns arrays: t, foot (1 left, 2 right), x_m, y_m, n_cells (per sample and foot); stance_t0, stance_t1,
    stance_foot, stance_x_m, stance_y_m; n_unlabelled_cells; format."""
    empty = {"t": np.zeros(0, np.float32), "foot": np.zeros(0, np.int8), "x_m": np.zeros(0, np.float32), "y_m": np.zeros(0, np.float32),
             "n_cells": np.zeros(0, np.int32), "stance_t0": np.zeros(0, np.float32), "stance_t1": np.zeros(0, np.float32), "stance_foot": np.zeros(0, np.int8),
             "stance_x_m": np.zeros(0, np.float32), "stance_y_m": np.zeros(0, np.float32), "n_unlabelled_cells": 0, "format": "none"}
    rows = []; n_other = 0; n = len(wx)
    legacy = n > 0 and not any("|" in str(v) for v in wx) and (wfoot is None or not any(_split(v) for v in wfoot))   # plain numbers, no labels anywhere
    fmt = "footfall_cm" if legacy else "cells"
    for i in range(n):
        xs = _split(wx[i])
        if not xs:
            continue
        ys = _split(wy[i]) if wy is not None and i < len(wy) else []
        if legacy:
            try:                                                                   # legacy: one footfall position per row, cm
                x = float(xs[0]); y = float(ys[0]) if ys else 0.0
            except ValueError:
                continue
            rows.append((float(t[i]), 0, x / 100.0, y / 100.0, 1)); continue
        m = min(len(xs), len(ys))
        if m < WALKWAY_MIN_CELLS:
            continue
        try:
            X = np.array(xs[:m], float); Y = np.array(ys[:m], float)
        except ValueError:
            continue
        labs = _split(wfoot[i]) if wfoot is not None and i < len(wfoot) else []
        pr = _split(wpress[i]) if wpress is not None and i < len(wpress) else []
        try:
            P = np.array(pr[:m], float) if len(pr) >= m else np.ones(m)
        except ValueError:
            P = np.ones(m)
        P = np.where(np.isfinite(P) & (P > 0), P, 1.0)
        L = np.array(labs[:m]) if len(labs) >= m else np.array([""] * m)
        for lab, code in (("L", 1), ("R", 2)):
            k = L == lab
            if k.sum() >= WALKWAY_MIN_CELLS:
                rows.append((float(t[i]), code, float((X[k] * P[k]).sum() / P[k].sum() * cell_m), float((Y[k] * P[k]).sum() / P[k].sum() * cell_m), int(k.sum())))
        n_other += int((~np.isin(L, ["L", "R"])).sum())
    if not rows:
        empty["n_unlabelled_cells"] = n_other; empty["format"] = fmt if n_other else "none"
        return empty
    rows.sort(); A = np.array(rows, float)
    out = {"t": A[:, 0].astype(np.float32), "foot": A[:, 1].astype(np.int8), "x_m": A[:, 2].astype(np.float32), "y_m": A[:, 3].astype(np.float32),
           "n_cells": A[:, 4].astype(np.int32), "n_unlabelled_cells": n_other, "format": fmt}
    # stances: consecutive samples of one foot whose centre stays within STANCE_DRIFT_M (the other foot's samples
    # interleave during double support, so each foot keeps its own open stance)
    st = []; cur = {}
    for tt, code, x, y, _ in rows:
        s = cur.get(code)
        if s is not None and tt - s["t1"] <= STANCE_GAP_S and abs(x - s["x"][0]) <= STANCE_DRIFT_M and abs(y - s["y"][0]) <= STANCE_DRIFT_M:
            s["t1"] = tt; s["x"].append(x); s["y"].append(y)
        else:
            cur[code] = {"foot": code, "t0": tt, "t1": tt, "x": [x], "y": [y]}; st.append(cur[code])
    st.sort(key=lambda s: s["t0"])
    if fmt == "footfall_cm":
        keep = st                                                                  # every legacy row is its own footfall
    else:
        keep = [s for s in st if s["t1"] - s["t0"] >= STANCE_MIN_S]
    out.update(stance_t0=np.array([s["t0"] for s in keep], np.float32), stance_t1=np.array([s["t1"] for s in keep], np.float32),
               stance_foot=np.array([s["foot"] for s in keep], np.int8), stance_x_m=np.array([np.median(s["x"]) for s in keep], np.float32),
               stance_y_m=np.array([np.median(s["y"]) for s in keep], np.float32))
    return out


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
        if c in RAW_COLS:
            out[c] = np.array(vals, dtype=object)
        else:
            out[c] = _safe_float(vals)
    return out


def make_imu_ref(csv_path: str | Path, out_path: str | Path, recording_id: str | None = None,
                 subject: str | None = None, fs: float = IMU_FS, walkway_cell_cm: float = WALKWAY_CELL_CM) -> dict:
    csv_path = Path(csv_path); out_path = Path(out_path)
    cols = ["Time", "GeneralEvent", f"{PELVIS_SENSOR}_FreeAcc_U"] + CONTACT_COLS + WALKWAY_COLS + WALKWAY_EXTRA
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
    ww = parse_walkway(t, data[WALKWAY_COLS[0]], data[WALKWAY_COLS[1]], data.get(WALKWAY_EXTRA[0]), data.get(WALKWAY_EXTRA[1]),
                       cell_m=float(walkway_cell_cm) / 100.0) if all(c in data for c in WALKWAY_COLS) else parse_walkway(t, [], [])
    if ww["format"] == "footfall_cm" and all(c in data for c in CONTACT_COLS):
        # legacy rows carry no foot: the contact flag that turned on within 80 ms of the row names it (1 left, 2 right)
        for lab, col in ((1, CONTACT_COLS[0]), (2, CONTACT_COLS[1])):
            cf = np.nan_to_num(_safe_float(data[col])) > 0.5; on_t = t[np.flatnonzero(cf & ~np.r_[False, cf[:-1]])]
            if len(on_t):
                near = np.abs(ww["t"][:, None] - on_t[None, :]).min(1) <= 0.08
                ww["foot"][near & (ww["foot"] == 0)] = lab; ww["foot"][near & (ww["foot"] != 0) & (ww["foot"] != lab)] = 0
        ww["stance_foot"] = ww["foot"].copy()
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
                        walkway_t=ww["t"], walkway_x_m=ww["x_m"], walkway_y_m=ww["y_m"], walkway_foot=ww["foot"], walkway_n_cells=ww["n_cells"],
                        walkway_stance_t0=ww["stance_t0"], walkway_stance_t1=ww["stance_t1"], walkway_stance_foot=ww["stance_foot"],
                        walkway_stance_x_m=ww["stance_x_m"], walkway_stance_y_m=ww["stance_y_m"], walkway_cell_m=float(walkway_cell_cm) / 100.0,
                        walkway_format=ww["format"], walkway_n_unlabelled_cells=int(ww["n_unlabelled_cells"]),
                        duration_s=float(grid[-1]), n_rows=int(n_rows_file), nan_fraction=nan_fraction,
                        source_sha256=_sha256(csv_path), source_name=csv_path.name, recording_id=rec, subject=subj,
                        schema=SCHEMA)
    walk_s = float(sum(b - a for l, a, b in zip(labels, t0s, t1s) if l == "Walk"))
    return {"recording_id": rec, "subject": subj, "duration_s": float(grid[-1]), "walk_s": walk_s,
            "nan_fraction": nan_fraction, "n_walkway_rows": int(len(ww["t"])), "n_walkway_stances": int(len(ww["stance_t0"])),
            "walkway_format": ww["format"], "out": str(out_path)}


def load_imu_ref(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    for k in ("fs", "duration_s", "nan_fraction", "contact_fraction"):
        if k in d:
            d[k] = float(d[k])
    for k in ("recording_id", "subject", "source_sha256", "source_name", "schema", "walkway_format"):
        if k in d:
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
            "walkway_stances": int(len(d.get("walkway_stance_t0", []))), "walkway_format": d.get("walkway_format", "none"),
            "nan_fraction": d["nan_fraction"], "source_sha256": d["source_sha256"], "source_name": d.get("source_name")}


def load_recording(csv_path: Path, cache_dir: Path, recording_id: str, subject: str, params: dict | None = None) -> dict:
    """Derived reference for a recording CSV, cached in cache_dir (re-made when the CSV, the schema or the walkway
    cell size changes)."""
    from .video_io import sha256_file
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{recording_id}_imu_ref.npz"
    cell_cm = float((params or {}).get("walkway_cell_cm", WALKWAY_CELL_CM))
    if out.exists():
        d = load_imu_ref(out)
        if d["source_sha256"] == sha256_file(csv_path) and str(d.get("schema")) == SCHEMA and abs(float(d.get("walkway_cell_m", -1)) * 100 - cell_cm) < 1e-6:
            return d
    make_imu_ref(csv_path, out, recording_id=recording_id, subject=subject, walkway_cell_cm=cell_cm)
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
