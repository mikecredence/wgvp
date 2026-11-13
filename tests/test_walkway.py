"""The pressure-walkway columns of a WearGait recording export, as the public files carry them: per IMU sample the
lists of loaded cells with foot labels and pressure levels. The reader must turn them into one footprint centre
per sample and foot, group the samples of one footprint into stances, ignore unlabelled cells, and still accept
the one-number-per-row legacy form."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.imu_ref import load_imu_ref, make_imu_ref, parse_walkway   # noqa: E402

CELL = 0.0127


def _footprint(cx, cy, label, n=40, length=14, width=8):
    """Cells of one footprint around (cx, cy) in cell units: a filled ellipse with the given extents."""
    xs, ys, labs, ps = [], [], [], []
    for dx in range(-length // 2, length // 2 + 1):
        for dy in range(-width // 2, width // 2 + 1):
            if (2 * dx / length) ** 2 + (2 * dy / width) ** 2 <= 1.0:
                xs.append(int(round(cx)) + dx); ys.append(int(round(cy)) + dy); labs.append(label); ps.append(2 + (abs(dx) + abs(dy)) % 6)
    return xs, ys, labs, ps


def _rows(fs=100.0):
    """A walk of 6 steps along the mat: left and right footprints 0.6 m apart, one every 0.6 s, each loaded for
    0.7 s (0.1 s of double support), plus an unlabelled blob (a cane) during the third step and an empty stretch
    before and after."""
    t = np.arange(int(8.0 * fs) + 1) / fs; X = [""] * len(t); Y = [""] * len(t); F = [""] * len(t); P = [""] * len(t)
    steps = [(1.0 + 0.6 * k, 1, 20 + 3 * (k % 2) * (-1) ** k, "L" if k % 2 == 0 else "R") for k in range(6)]    # (t_on, _, y_cells, label)
    truth = []
    for k, (t_on, _, yc, lab) in enumerate(steps):
        xc = (0.5 + 0.6 * k) / CELL; truth.append((lab, t_on, t_on + 0.7, xc * CELL, yc * CELL))
        for i in np.flatnonzero((t >= t_on) & (t < t_on + 0.7)):
            xs, ys, ls, ps = _footprint(xc, yc, lab)
            if k == 2:
                xs2, ys2, ls2, ps2 = _footprint(xc + 30, yc + 15, "Other 1", length=4, width=4); xs += xs2; ys += ys2; ls += ls2; ps += ps2
            X[i] = (X[i] + "|" if X[i] else "") + "|".join(map(str, xs)); Y[i] = (Y[i] + "|" if Y[i] else "") + "|".join(map(str, ys))
            F[i] = (F[i] + "|" if F[i] else "") + "|".join(ls); P[i] = (P[i] + "|" if P[i] else "") + "|".join(map(str, ps))
    return t, X, Y, F, P, truth


def test_parse_cells_into_footprints_and_stances():
    t, X, Y, F, P, truth = _rows()
    w = parse_walkway(t, X, Y, F, P, cell_m=CELL)
    assert w["format"] == "cells" and len(w["stance_t0"]) == 6 and w["n_unlabelled_cells"] > 0
    for (lab, t0, t1, x, y), f, s0, s1, sx, sy in zip(truth, w["stance_foot"], w["stance_t0"], w["stance_t1"], w["stance_x_m"], w["stance_y_m"]):
        assert f == (1 if lab == "L" else 2) and abs(s0 - t0) < 0.011 and abs(s1 - (t1 - 0.01)) < 0.011
        assert abs(sx - x) < 0.6 * CELL and abs(sy - y) < 0.6 * CELL, (sx, x, sy, y)      # the centre of a symmetric footprint is its middle cell
    # per-sample rows: one per loaded foot, two during the double support
    n_double = int(np.sum((t >= 1.6) & (t < 1.7))); assert np.sum((w["t"] >= 1.6) & (w["t"] < 1.7)) == 2 * n_double


def test_make_imu_ref_reads_the_export_form(tmp_path):
    t, X, Y, F, P, truth = _rows()
    rng = np.random.default_rng(0); csv_path = tmp_path / "S9_selfpace_mat.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow(["Time", "GeneralEvent", "L Foot Contact", "R Foot Contact", "Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot", "LowerBack_FreeAcc_U"])
        for i, ti in enumerate(t):
            wr.writerow([f"{ti:.2f} sec", "Walk", 0, 0, X[i], Y[i], P[i], F[i], f"{rng.normal(0, 0.5):.3f}"])
    info = make_imu_ref(csv_path, tmp_path / "ref.npz", recording_id="S9_selfpace_mat", subject="S9")
    d = load_imu_ref(tmp_path / "ref.npz")
    assert info["walkway_format"] == "cells" and info["n_walkway_stances"] == 6 and len(d["walkway_t"]) == info["n_walkway_rows"] > 250
    assert abs(float(d["walkway_cell_m"]) - CELL) < 1e-9 and d["walkway_stance_foot"].tolist() == [1, 2, 1, 2, 1, 2]
    # a different cell size scales the positions
    make_imu_ref(csv_path, tmp_path / "ref2.npz", recording_id="S9_selfpace_mat", subject="S9", walkway_cell_cm=2.54)
    d2 = load_imu_ref(tmp_path / "ref2.npz"); assert np.allclose(d2["walkway_stance_x_m"], 2 * d["walkway_stance_x_m"], atol=1e-5)


def test_legacy_footfall_rows_still_read():
    t = np.arange(0, 5.0, 0.01); X = [""] * len(t); Y = [""] * len(t)
    for k, ti in enumerate([0.5, 1.1, 1.7, 2.3]):
        i = int(round(ti * 100)); X[i] = str(60 + 55 * k); Y[i] = str(20 + 10 * (k % 2))
    w = parse_walkway(t, X, Y, None, None)
    assert w["format"] == "footfall_cm" and len(w["stance_t0"]) == 4 and abs(w["stance_x_m"][1] - 1.15) < 1e-6 and (w["stance_foot"] == 0).all()
