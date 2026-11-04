"""Sync of the combined video to the recording timeline. The two views are one file, so one offset per
(recording, video): the video PTS time of IMU t=0.

primary      pelvis cross-correlation (joint over both views) - from selection
cross-checks light flash (if a sync light is visible), Walk-interval correlation against an exported label file
             (Video Labeler scene labels in video time), annotation-sheet start frame
"""
from __future__ import annotations

import numpy as np

from .flash import nearest_flash


def _frame_at(pts_ms: np.ndarray, frame_idx: np.ndarray, t_s: float) -> int:
    return int(frame_idx[int(np.argmin(np.abs(pts_ms / 1000.0 - t_s)))])


def combine_sync(primary: dict, flash_cands: list[dict], labels: dict | None, sheet_frame: int | None, frame_idx: np.ndarray,
                 pts_ms: np.ndarray, fps: float, tol_frames: float = 3.0, flash_search_s: float = 3.0, imu_duration_s: float | None = None,
                 min_overlap_frac: float = 0.6, strong_flash_z: float = 100.0) -> dict:
    """Combine the pelvis cross-correlation with the cross-checks into one sync record. Two sanity rules guard against a
    confidently wrong cross-correlation (seen on a rendered walker who paced within a metre with a 5-mm pelvis bob: the
    side view fragmented, the fragments' own lags disagreed, and the joint lag landed 30 s off with r 0.39): the offset
    must leave at least `min_overlap_frac` of the recording inside the video, and a strong persistent light-like
    brightness step (z >= strong_flash_z) that sits more than `flash_search_s` from the offset is reported as a
    disagreement even though it was not used."""
    tol_s = tol_frames / fps
    x_ok = bool(primary.get("ok")) and primary.get("offset_s") is not None
    x_off = float(primary["offset_s"]) if x_ok else None
    x_conf = primary.get("confidence", "none") if x_ok else "none"
    rec = {"xcorr": {"available": x_ok, "offset_s": x_off, "confidence": x_conf, "r_fine": primary.get("r_fine"), "margin": primary.get("margin")},
           "flash": {"available": bool(flash_cands), "n_candidates": len(flash_cands), "candidates": flash_cands[:5], "used": None},
           "labels": labels or {"available": False},
           "sheet": {"available": sheet_frame is not None, "frame": sheet_frame,
                     "offset_s": float(pts_ms[np.clip(np.searchsorted(frame_idx, sheet_frame), 0, len(frame_idx) - 1)] / 1000.0) if sheet_frame is not None and len(frame_idx) else None},
           "tolerance_frames": tol_frames}
    anchor = x_off if x_ok else (labels or {}).get("offset_s") if (labels or {}).get("available") else (rec["sheet"]["offset_s"] if rec["sheet"]["available"] else None)
    fl = None
    if flash_cands:
        fl = nearest_flash(flash_cands, anchor, flash_search_s) if anchor is not None else (flash_cands[0] if len(flash_cands) == 1 else None)
        if fl is None:
            rec["flash"]["note"] = "no flash onset near the primary offset (the light may not be in view: it is only a cross-check)"
    rec["flash"]["used"] = fl
    agree = {}
    def cmp(name, a, b):
        if a is not None and b is not None:
            agree[f"{name}_s"] = float(a - b); agree[name] = bool(abs(a - b) <= tol_s)
    cmp("flash_minus_xcorr", fl["onset_t_s"] if fl else None, x_off)
    cmp("labels_minus_xcorr", (labels or {}).get("offset_s") if (labels or {}).get("available") else None, x_off)
    cmp("sheet_minus_xcorr", rec["sheet"]["offset_s"] if rec["sheet"]["available"] else None, x_off)
    rec["agreement"] = agree
    flags = []
    if x_ok:
        chosen, src = x_off, "pelvis_xcorr"
        status = "PASS" if x_conf == "high" else "WARN"
        if x_conf != "high":
            flags.append("xcorr_low_confidence")
        checks = [k for k in ("flash_minus_xcorr", "labels_minus_xcorr", "sheet_minus_xcorr") if k in agree]
        if any(agree[k] for k in checks):
            src += "(validated by " + ",".join(k.split("_")[0] for k in checks if agree[k]) + ")"
        for k in checks:
            if not agree[k]:
                flags.append(f"{k.split('_')[0]}_disagrees ({agree[k + '_s']:+.3f}s)"); status = "WARN" if status == "PASS" else status
        if imu_duration_s and len(pts_ms):
            v0, v1 = float(pts_ms[0] / 1000.0), float(pts_ms[-1] / 1000.0)
            inside = max(0.0, min(v1, x_off + imu_duration_s) - max(v0, x_off)) / imu_duration_s
            rec["recording_inside_video_frac"] = inside
            if inside < min_overlap_frac:
                flags.append(f"offset leaves only {inside:.0%} of the recording inside the video"); status = "WARN"
        if flash_cands and fl is None:
            strong = max(flash_cands, key=lambda c: c.get("z", 0.0))
            if strong.get("z", 0.0) >= strong_flash_z and strong.get("persistent", True):
                agree["strong_light_step_minus_xcorr_s"] = float(strong["onset_t_s"] - x_off)
                flags.append(f"a strong light-like step at {strong['onset_t_s']:.2f}s ({strong.get('view', '?')} view) disagrees with the offset by {strong['onset_t_s'] - x_off:+.1f}s"); status = "WARN"
    elif (labels or {}).get("available"):
        chosen, src, status = labels["offset_s"], "walk_labels", "WARN"; flags.append("no_xcorr_subject")
    elif fl:
        chosen, src, status = fl["onset_t_s"], "flash", "WARN"; flags.append("no_xcorr_subject")
    elif rec["sheet"]["available"]:
        chosen, src, status = rec["sheet"]["offset_s"], "sheet", "WARN"; flags.append("no_xcorr_subject")
    else:
        chosen, src, status = None, "none", "FAIL"; flags.append("no_sync_source")
    rec.update(offset_s=chosen, offset_frame=_frame_at(pts_ms, frame_idx, chosen) if chosen is not None else None, source=src, status=status, flags=flags,
               message={"PASS": "sync from the pelvis cross-correlation", "WARN": "sync established with low confidence, from a cross-check only, or with a disagreement - "
                        "inspect agreement{} and the thumbnails", "FAIL": "no sync: the IMU cross-correlation found no subject and no label export / flash / sheet frame was available"}[status])
    return rec
