"""The fabricated example recording round-trips through the pipeline's IMU reader: the CSV written by
examples/fetch_examples.py yields a pelvis displacement in phase with the signal it was built from."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.fetch_examples import FS, acceleration_from_displacement, write_csv   # noqa: E402
from wgvp.imu_ref import make_imu_ref, load_imu_ref       # noqa: E402
from wgvp.signals import bandpass                         # noqa: E402


def test_fabricated_csv_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(int(30 * FS) + 1) / FS
    disp = 0.03 * np.sin(2 * np.pi * 1.9 * t) + 0.01 * np.sin(2 * np.pi * 0.95 * t + 0.4)     # step + stride harmonics
    inside = (t >= 4.0) & (t <= 26.0)
    disp = disp + 0.004 * rng.standard_normal(len(t)) - 2.0  # keypoint-like jitter; a non-zero mean like the real signal
    rec = {"t": t, "acc_up": acceleration_from_displacement(np.where(inside, disp, np.nan), inside, 0.5, rng),
           "acc_en": 0.3 * rng.standard_normal((len(t), 2)), "label": np.where(inside, "Walk", "Standing").astype(object)}
    csv = tmp_path / "EX99_selfpace.csv"; write_csv(csv, rec)
    info = make_imu_ref(csv, tmp_path / "ref.npz", recording_id="EX99_selfpace", subject="EX99")
    ref = load_imu_ref(tmp_path / "ref.npz")
    assert abs(ref["duration_s"] - 30.0) < 0.02 and info["walk_s"] > 21.0
    assert [e[0] for e in ref["events"]] == ["Standing", "Walk", "Standing"]
    w = (ref["t"] >= 6) & (ref["t"] <= 24)
    a = bandpass(ref["disp_up"].astype(float), FS)[w]; b = bandpass(disp, FS)[w]
    r = np.corrcoef(a, b)[0, 1]
    assert r > 0.9, f"recovered displacement not in phase with the planted one (r={r:.2f})"
