"""Two sanity rules of the sync combination, from a rendered case where the pelvis cross-correlation landed 30 s off
with a 'high' confidence: the offset must leave most of the recording inside the video, and a strong persistent
light-like step far from the offset is a disagreement even when it was not used."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.sync import combine_sync   # noqa: E402

FI = np.arange(1600); PTS = FI / 30.0 * 1000.0      # a 53-s video
LIGHT = [{"onset_t_s": 3.0, "onset_frame": 90, "z": 520.0, "persistent": True, "view": "right", "step": 100.0}]


def test_offset_outside_the_video_is_flagged():
    s = combine_sync({"ok": True, "offset_s": -27.4, "confidence": "high", "r_fine": 0.39, "margin": 0.24}, [], None, None, FI, PTS, 30.0, imu_duration_s=48.5)
    assert s["status"] == "WARN" and any("inside the video" in f for f in s["flags"]) and s["recording_inside_video_frac"] < 0.5


def test_strong_far_light_step_is_a_disagreement():
    s = combine_sync({"ok": True, "offset_s": -27.4, "confidence": "high"}, LIGHT, None, None, FI, PTS, 30.0)
    assert s["status"] == "WARN" and any("light-like step" in f for f in s["flags"]) and abs(s["agreement"]["strong_light_step_minus_xcorr_s"] - 30.4) < 0.01


def test_consistent_sync_stays_pass():
    s = combine_sync({"ok": True, "offset_s": 3.02, "confidence": "high"}, LIGHT, None, None, FI, PTS, 30.0, imu_duration_s=48.5)
    assert s["status"] == "PASS" and s["flags"] == [] and "validated by flash" in s["source"] and s["recording_inside_video_frac"] > 0.95
    weak = [{"onset_t_s": 20.0, "onset_frame": 600, "z": 15.0, "persistent": True, "view": "left", "step": 30.0}]   # a small step elsewhere: not raised
    s = combine_sync({"ok": True, "offset_s": 3.02, "confidence": "high"}, weak, None, None, FI, PTS, 30.0, imu_duration_s=48.5)
    assert s["status"] == "PASS" and s["flags"] == []
