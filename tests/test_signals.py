"""Unit tests for the CPU core: cross-correlation convention, offset recovery with lag / dropped frames / decoys,
flash detection, selection decision rule, sync rule, reconciliation and the privacy check. Run anywhere:
    python -m pytest scripts/wg_video_pipeline/tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.signals import masked_xcorr, match_signals, bandpass, rms_envelope, double_integrate  # noqa: E402
from wgvp.flash import _step_candidates, detect_flash_auto  # noqa: E402
from wgvp.selection import decide_view  # noqa: E402
from wgvp.sync import combine_sync  # noqa: E402
from wgvp.reconcile import signal_agreement, reconcile_recording  # noqa: E402

FS = 100.0


def gait_like(rng, dur=50.0, f_step=2.0, bursts=((5, 20), (26, 45)), noise=0.05):
    t = np.arange(0, dur, 1 / FS)
    act = np.zeros_like(t, bool)
    for a, b in bursts:
        act |= (t > a) & (t < b)
    env = np.convolve(act.astype(float), np.ones(50) / 50, mode="same")
    x = env * (np.sin(2 * np.pi * f_step * t + 0.2 * np.sin(2 * np.pi * 0.07 * t)) + 0.4 * np.sin(2 * np.pi * f_step / 2 * t + 0.3))
    return t, x + noise * rng.standard_normal(len(t))


def test_xcorr_lag_sign():
    rng = np.random.default_rng(1)
    b = rng.standard_normal(2000); k = 37
    a = np.roll(b, k); a[:k] = np.nan
    lags, r, n = masked_xcorr(a, b, 200)
    assert lags[np.nanargmax(r)] == k and np.nanmax(r) > 0.99


def test_xcorr_edge_overlap_not_inflated():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(500); b = rng.standard_normal(500)
    lags, r, n = masked_xcorr(a, b, 480, min_overlap=20)
    assert np.nanmax(np.abs(r[n < 60])) < 0.85   # small overlaps are true Pearson (chance level for n~20), not ~1
    assert np.nanmax(np.abs(r[n >= 200])) < 0.35


@pytest.mark.parametrize("offset,drop", [(7.3, 0), (-3.0, 0), (12.345, 40), (0.0, 10), (25.0, 0)])
def test_offset_recovery(offset, drop):
    rng = np.random.default_rng(3)
    t, u = gait_like(rng)
    tv = np.arange(0, 80, 1 / 30.0) + rng.normal(0, 0.0005, 2400)
    v = np.interp(tv - offset, t, u, left=0, right=0) + 0.3 * rng.standard_normal(len(tv))
    keep = np.ones(len(tv), bool)
    if drop:
        keep[rng.choice(len(tv), drop, replace=False)] = False
    res = match_signals(tv[keep], v[keep], u)
    assert res["ok"] and abs(res["offset_s"] - offset) < 0.02 and res["r_fine"] > 0.8


def test_decoy_same_activity_scores_low():
    rng = np.random.default_rng(4)
    t, u = gait_like(rng)
    _, u2 = gait_like(rng, f_step=1.9)
    tv = np.arange(0, 70, 1 / 30.0)
    v_true = np.interp(tv - 7.3, t, u, left=0, right=0) + 0.3 * rng.standard_normal(len(tv))
    v_decoy = np.interp(tv - 7.3, t, u2, left=0, right=0) + 0.3 * rng.standard_normal(len(tv))
    r_true = match_signals(tv, v_true, u); r_dec = match_signals(tv, v_decoy, u)
    assert r_true["r_fine"] - r_dec["r_fine"] > 0.4
    assert r_dec["r_env_coarse"] > 0.8   # same activity pattern: the envelope alone cannot tell them apart


def test_double_integrate_in_phase():
    t = np.arange(0, 30, 1 / FS); disp = np.sin(2 * np.pi * 2.0 * t)
    acc = -(2 * np.pi * 2.0) ** 2 * disp
    d = double_integrate(acc, FS)
    r = np.corrcoef(d[500:-500], disp[500:-500])[0, 1]
    assert r > 0.98


def test_flash_step_detector():
    rng = np.random.default_rng(5)
    s = 40 + rng.normal(0, 1.5, 3000); s[1234:2500] += 180
    c = _step_candidates(s, 30.0)
    assert c and abs(c[0]["onset_frame_pos"] - 1234) <= 1 and c[0]["persistent"]
    # a person walking through (transient bump) must not win
    s2 = 40 + rng.normal(0, 1.5, 3000); s2[800:812] += 60
    c2 = [x for x in _step_candidates(s2, 30.0) if x["persistent"]]
    assert not c2


def test_flash_auto_grid():
    rng = np.random.default_rng(6)
    F = 900; grid = (60 + rng.normal(0, 2, (F, 18, 32))).astype(np.float32)
    grid[300:, 4, 7] += 150                                     # light at cell (7, 4)
    grid += np.linspace(0, 25, F)[:, None, None]                # global exposure drift
    c = detect_flash_auto(grid.clip(0, 255).astype(np.uint8), np.arange(F), np.arange(F) * 1000 / 30, 30.0)
    assert c and c[0]["onset_frame"] == 300 and c[0]["cell"] == [7, 4]


def test_decide_rule():
    P = {"confidence": "high", "track_ids": [3], "r_fine": 0.7, "margin": 0.4}
    R = {"ok": True, "track_ids": [3]}; K = {"ok": False, "track_ids": []}
    d = decide_view(P, "high", R, K); assert d["status"] == "PASS" and d["track_ids"] == [3] and d["agreement"]["primary_vs_roi"]
    d = decide_view({**P, "confidence": "low"}, "low", {"ok": False, "track_ids": []}, {"ok": True, "track_ids": [5]})
    assert d["status"] == "WARN" and d["track_ids"] == [5] and d["source"] == "operator_pick"
    d = decide_view({"confidence": "none", "track_ids": []}, "none", {"ok": False, "track_ids": []}, K)
    assert d["status"] == "FAIL" and "pick-subject" in d["message"]
    # the other view carries the lag with high confidence; this view is weak -> WARN but chosen
    d = decide_view({"confidence": "low", "track_ids": [7]}, "high", {"ok": False, "track_ids": []}, K)
    assert d["status"] == "WARN" and d["track_ids"] == [7]


def test_sync_rule():
    fi = np.arange(3000); pts = fi * 1000 / 30
    flash = [{"onset_frame": 90, "onset_t_s": 3.0, "z": 50}]
    prim = {"ok": True, "offset_s": 3.02, "confidence": "high", "r_fine": 0.7, "margin": 0.3}
    s = combine_sync(prim, flash, None, None, fi, pts, 30.0)
    assert s["status"] == "PASS" and s["source"].startswith("pelvis_xcorr(validated by flash") and s["offset_frame"] == 91
    s = combine_sync({"ok": True, "offset_s": 5.0, "confidence": "high"}, flash, None, None, fi, pts, 30.0)
    assert s["status"] == "WARN" and s["source"] == "pelvis_xcorr" and any(f.startswith("flash_disagrees") for f in s["flags"])
    s = combine_sync({"ok": False}, [], None, 120, fi, pts, 30.0)
    assert s["source"] == "sheet" and s["offset_frame"] == 120
    lab = {"available": True, "offset_s": 3.0, "r": 0.9}
    s = combine_sync(prim, [], lab, None, fi, pts, 30.0)
    assert s["status"] == "PASS" and "labels" in s["source"]
    s = combine_sync({"ok": False}, [], None, None, fi, pts, 30.0)
    assert s["status"] == "FAIL"


def test_reconcile_two_cameras():
    rng = np.random.default_rng(7)
    t, u = gait_like(rng)
    tA = np.arange(0, 48, 1 / 30); tB = np.arange(0, 48, 1 / 25) + 0.011
    xA = np.interp(tA, t, u) + 0.2 * rng.standard_normal(len(tA)); xB = np.interp(tB, t, u) + 0.2 * rng.standard_normal(len(tB))
    sg = signal_agreement(tA, xA, tB, xB)
    assert sg["ok"] and abs(sg["residual_lag_s"]) < 0.02 and sg["r_best"] > 0.8
    cams = {"cam1": {"fps": 30, "sync": {"agreement": {"flash_minus_xcorr_s": 0.05}}, "hip": (tA, xA), "pose": None},
            "cam2": {"fps": 25, "sync": {"agreement": {"flash_minus_xcorr_s": 0.06}}, "hip": (tB, xB), "pose": None}}
    rec = reconcile_recording(cams)
    assert rec["status"] == "PASS" and abs(rec["pairs"]["cam1|cam2"]["flash_latency_diff_s"] + 0.01) < 1e-9


def test_privacy_check(tmp_path):
    from wgvp.qc import privacy_check
    d = tmp_path / "deliver"; (d / "validation_bundle").mkdir(parents=True)
    np.savez(d / "ok.npz", a=np.zeros(10, np.float32)); (d / "x.json").write_text("{}")
    assert privacy_check(d)["ok"]
    import cv2
    img = np.zeros((100, 100, 3), np.uint8); cv2.line(img, (0, 0), (99, 99), (0, 255, 0), 2); cv2.imwrite(str(d / "validation_bundle" / "s.png"), img)
    assert privacy_check(d)["ok"]
    cv2.imwrite(str(d / "validation_bundle" / "photo.png"), np.full((100, 100, 3), 120, np.uint8))
    assert not privacy_check(d)["ok"]
    (d / "validation_bundle" / "photo.png").unlink(); (d / "clip.mp4").write_bytes(b"x")
    assert not privacy_check(d)["ok"]


def test_walk_label_offset():
    from wgvp.labels import walk_label_offset
    imu_events = [("Standing", 0, 5), ("Walk", 5, 15), ("Turn", 15, 17), ("Walk", 17, 27), ("Standing", 27, 40)]
    video_walk = [(5 + 7.25, 15 + 7.25), (17 + 7.25, 27 + 7.25)]
    r = walk_label_offset(video_walk, imu_events, 60.0, 40.0)
    assert r["available"] and abs(r["offset_s"] - 7.25) < 0.06 and r["r"] > 0.9


def _smooth_texture(rng, h, w, sigma=6):
    import cv2
    t = rng.normal(0, 1, (h, w)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), sigma); t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    yy = np.linspace(0.2, 1.0, h)[:, None]
    return np.clip(60 + 150 * (0.5 * t + 0.5 * yy), 0, 255).astype(np.uint8)


def test_panel_detection(tmp_path):
    """Combined frames like the real sessions: unequal panels, the right one letterboxed; a stacked variant too."""
    import cv2
    from wgvp.video_io import detect_panels
    rng = np.random.default_rng(8)
    W, H, LW, RC = 1420, 600, 520, 500
    texA = _smooth_texture(rng, H + 40, LW); texB = _smooth_texture(rng, RC, W - LW + 40)
    out = tmp_path / "combined.mp4"; w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in range(90):
        fr = np.zeros((H, W, 3), np.uint8)
        dy = int(round(10 * np.sin(f / 9))); dx = int(round(12 * np.cos(f / 11)))
        fr[:, :LW] = texA[20 + dy:20 + dy + H][..., None]; fr[:RC, LW:] = texB[:, 20 + dx:20 + dx + W - LW][..., None]
        # a walking "person" (dark blob) crossing the left view
        cv2.circle(fr, (int(60 + 4 * f), 300), 25, (40, 40, 40), -1)
        w.write(fr)
    w.release()
    pan = detect_panels(out, n_samples=20)
    boxes = {p["id"]: p["box"] for p in pan["panels"]}
    assert pan["layout"] == "horizontal", pan
    assert abs(boxes["left"][2] - LW) <= 12 and abs(boxes["right"][0] - LW) <= 12 and abs(boxes["right"][3] - RC) <= 12, boxes
    # stacked variant: top panel full width 300 rows, bottom panel 280 rows with black bars left/right
    out2 = tmp_path / "stacked.mp4"; w = cv2.VideoWriter(str(out2), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    texC = _smooth_texture(rng, 300 + 40, W); texD = _smooth_texture(rng, 280, W - 300 + 40)
    for f in range(90):
        fr = np.zeros((H, W, 3), np.uint8)
        dy = int(round(10 * np.sin(f / 9))); dx = int(round(12 * np.cos(f / 11)))
        fr[:300, :] = texC[20 + dy:20 + dy + 300][..., None]; fr[320:, 150:W - 150] = texD[:, 20 + dx:20 + dx + W - 300][..., None]
        w.write(fr)
    w.release()
    pan2 = detect_panels(out2, n_samples=20); b2 = {p["id"]: p["box"] for p in pan2["panels"]}
    assert pan2["layout"] == "vertical", pan2
    assert abs(b2["top"][3] - 300) <= 14 and abs(b2["bottom"][1] - 320) <= 14 and abs(b2["bottom"][0] - 150) <= 14, b2
    # a single view with letterbox bars must NOT be split
    out3 = tmp_path / "single.mp4"; w = cv2.VideoWriter(str(out3), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    texE = _smooth_texture(rng, 520, W)
    for f in range(60):
        fr = np.zeros((H, W, 3), np.uint8); dy = int(round(10 * np.sin(f / 9)))
        fr[60:540, :] = texE[20 + dy:20 + dy + 480][..., None]; w.write(fr)
    w.release()
    pan3 = detect_panels(out3, n_samples=20)
    assert pan3["layout"] == "single" and abs(pan3["panels"][0]["box"][1] - 60) <= 12, pan3


def test_single_view_with_a_wall_corner_is_not_split(tmp_path):
    """One still camera, a hard vertical scene edge mid-frame (a wall corner) and a walker crossing it: one view, and
    its edge-map preview passes the privacy check even though the whole frame is textured."""
    import cv2
    from wgvp.qc import is_skeleton_only
    from wgvp.video_io import detect_panels, panel_preview
    rng = np.random.default_rng(11)
    W, H, C = 1280, 720, 700
    bg = np.empty((H, W), np.uint8); bg[:, :C] = 70; bg[:, C:] = 190
    bg = bg + rng.normal(0, 12, (H, W))                                          # grass-like grain everywhere
    out = tmp_path / "corner.mp4"; w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in range(240):
        fr = np.repeat(np.clip(bg + rng.normal(0, 14, (H, W)), 0, 255).astype(np.uint8)[..., None], 3, 2)   # sensor noise
        x = int(C + 160 * np.sin(2 * np.pi * f / 240)); cv2.rectangle(fr, (x - 40, 300), (x + 40, 440), (130, 130, 130), -1)
        w.write(fr)
    w.release()
    pan = detect_panels(out)
    assert pan["detail"]["score"] >= 1.6, pan          # the corner alone scores as a seam
    assert pan["layout"] == "single" and pan["detail"].get("vetoed"), pan
    png = panel_preview(out, pan["panels"], tmp_path / "preview.png", mode="edges")
    assert is_skeleton_only(Path(png))


def test_skeleton_only_accepts_dense_renders_and_rejects_photos(tmp_path):
    """The privacy check must pass a crowd of skeletons (many lines, few colours) and fail anything photographic."""
    import cv2
    from wgvp.qc import is_skeleton_only
    rng = np.random.default_rng(0)
    crowd = np.zeros((360, 640, 3), np.uint8)
    for _ in range(60):                                   # 60 skeleton-like figures: far denser than 8 % non-black
        x, y = rng.integers(10, 600), rng.integers(10, 300)
        col = (60, 220, 60) if rng.random() < 0.3 else (150, 150, 150)
        for dx, dy in ((0, 40), (-12, 20), (12, 20), (-8, 60), (8, 60)):
            cv2.line(crowd, (int(x), int(y)), (int(x + dx), int(y + dy)), col, 3)
        cv2.rectangle(crowd, (int(x) - 15, int(y) - 5), (int(x) + 15, int(y) + 65), col, 1)
    assert (crowd.max(axis=2) > 20).mean() > 0.08
    cv2.imwrite(str(tmp_path / "crowd.png"), crowd); assert is_skeleton_only(tmp_path / "crowd.png")
    photo = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    photo = cv2.GaussianBlur(photo, (0, 0), 3)
    cv2.imwrite(str(tmp_path / "photo.png"), photo); assert not is_skeleton_only(tmp_path / "photo.png")
    dark_photo = (photo * 0.3).astype(np.uint8)          # a dim photograph is still a photograph
    cv2.imwrite(str(tmp_path / "dark.png"), dark_photo); assert not is_skeleton_only(tmp_path / "dark.png")



def test_pelvis_signal_survives_a_cut_head():
    """A walker filmed from the end of the walkway: as the walker approaches, the head and shoulders leave the frame
    and the box top sticks to the frame edge, so box height no longer tracks body size. With the thigh as scale
    reference the pelvis signal stays the one a full view would give."""
    from wgvp.tracking import hip_vertical_signal
    rng = np.random.default_rng(1); fps = 30.0; T = 300; t = np.arange(T) / fps
    scale = 100.0 * (1 + 3.0 * t / t[-1])                         # apparent size grows 4x as the walker approaches
    bob = 0.03 * np.sin(2 * np.pi * 1.9 * t)                      # pelvis vertical motion, 3 % of body scale
    cy = 120.0; hip_y = cy + 0.25 * scale - bob * scale + rng.normal(0, 0.5, T)
    def track(shoulders_visible, truncated_box):
        kp = np.zeros((T, 17, 3), np.float32)
        kp[:, 11, :] = np.c_[200 - 0.1 * scale, hip_y, np.ones(T)]; kp[:, 12, :] = np.c_[200 + 0.1 * scale, hip_y, np.ones(T)]
        kp[:, 13, :] = np.c_[200 - 0.1 * scale, hip_y + 0.44 * scale, np.ones(T)]; kp[:, 14, :] = np.c_[200 + 0.1 * scale, hip_y + 0.44 * scale, np.ones(T)]
        kp[:, 5, :] = np.c_[200 - 0.15 * scale, hip_y - 0.5 * scale, np.ones(T) * shoulders_visible]; kp[:, 6, :] = np.c_[200 + 0.15 * scale, hip_y - 0.5 * scale, np.ones(T) * shoulders_visible]
        top = hip_y - 0.9 * scale
        if truncated_box:
            top = np.maximum(top, 0.0)                              # the frame edge (row 0) cuts the box for most of the approach
        box = np.c_[200 - 0.2 * scale, top, 200 + 0.2 * scale, hip_y + 0.9 * scale]
        return {"kp": kp, "box": box, "pts_ms": t * 1000.0}
    assert (np.maximum(hip_y - 0.9 * scale, 0.0) == 0.0).mean() > 0.5          # the scenario really truncates
    _, full = hip_vertical_signal(track(1.0, False))                # reference: shoulders seen, box whole
    _, cut = hip_vertical_signal(track(0.0, True))                  # shoulders gone, box truncated: thigh must take over
    ok = np.isfinite(full) & np.isfinite(cut)
    assert ok.sum() > 250
    from wgvp.signals import bandpass
    r = np.corrcoef(bandpass(full[ok], fps), bandpass(cut[ok], fps))[0, 1]      # the gait-band oscillation must survive
    assert r > 0.95, r
