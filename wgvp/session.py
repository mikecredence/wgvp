"""Orchestration over a data root: every session found by the scan is processed (CPU stages here); the GPU stage
runs as a subprocess in the GVHMR environment when configured; then the cross-session QC summary is built.

Per video: panel detection -> one decode pass with per-view tracking (cached by file hash).
Per recording: for every candidate video, the joint two-view subject selection against the recording CSV; the video
with the best pelvis match holds the recording; sync; per-view lifter inputs (cropped chunks + subject boxes);
coverage / visibility QC; reconciliation between the two views.
Outputs: <root>/wgvp_output/<session_id>/{work,deliver,config}; only deliver/ (and the QC summary) come back.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .config import load_config
from .discovery import scan_root
from .flash import detect_flash_auto, detect_flash_roi
from .imu_ref import load_recording, summarize_imu_ref
from .labels import read_walk_intervals, walk_label_offset
from .qc import dump_json, privacy_check, tee_logger, write_thumbnails
from .reconcile import reconcile_recording
from .selection import decide_view, joint_selection, merged_subject_track, pick_selection, roi_selection
from .sync import combine_sync
from .tracking import TrackSet, analyze_video, hip_vertical_signal, split_fragments
from .video_io import cut_video, detect_panels, panel_preview, probe_video, sha256_file

PKG_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PKG_DIR / "weights" / "yolov8n-pose.pt"
OUT_DIRNAME = "wgvp_output"


def out_root(root: Path) -> Path:
    return Path(root) / OUT_DIRNAME


def session_out(root: Path, session_id: str) -> Path:
    return out_root(root) / session_id


def environment_record() -> dict:
    env = {"wgvp_version": __version__, "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    for m in ("numpy", "scipy", "cv2", "torch", "ultralytics", "imageio_ffmpeg"):
        try:
            mod = __import__(m); env[m] = getattr(mod, "__version__", "?")
        except Exception:
            env[m] = None
    try:
        import torch
        env["cuda"] = bool(torch.cuda.is_available()); env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["cuda"] = None
    from .video_io import ffmpeg_exe
    env["ffmpeg"] = ffmpeg_exe()
    return env


def _load_session_config(cdir: Path) -> dict:
    cfg = {}
    if cdir.exists():
        for f in cdir.glob("*.json"):
            try:
                cfg[f.stem] = json.loads(f.read_text())
            except Exception:
                pass
    return cfg


def _status_merge(*sts):
    sts = [s for s in sts if s]
    if "FAIL" in sts:
        return "FAIL"
    if "WARN" in sts:
        return "WARN"
    return "PASS"


# ----------------------------------------------------------------------------------------------------------------
def panels_for(video: Path, cfg: dict, work: Path, log) -> dict:
    ov = cfg["views"].get("override", {}).get(video.name)
    if ov:
        pan = {"layout": "override", "panels": [{"id": k, "box": [int(x) for x in v], "width": int(v[2] - v[0]), "height": int(v[3] - v[1])} for k, v in ov.items()], "seam": None}
    elif cfg["views"]["layout"] == "single":
        info = probe_video(video, with_hash=False)
        pan = {"layout": "single", "panels": [{"id": "full", "box": [0, 0, info["width"], info["height"]], "width": info["width"], "height": info["height"]}], "seam": None}
    else:
        pan = detect_panels(video, n_samples=cfg["views"]["n_sample_frames"], black_level=cfg["views"]["black_level"], min_panel_frac=cfg["views"]["min_panel_frac"])
        if cfg["views"]["layout"] in ("horizontal", "vertical") and pan["layout"] != cfg["views"]["layout"]:
            pan["flag"] = f"config asks for {cfg['views']['layout']} but {pan['layout']} was detected; using the detection - set views.override to force boxes"
    work.mkdir(parents=True, exist_ok=True)
    try:
        pan["preview_png"] = panel_preview(video, pan["panels"], work / "panels_preview.png", mode="edges")
    except Exception as e:
        pan["preview_error"] = str(e)
    dump_json(pan, work / "panels.json")
    log(f"    panels: {pan['layout']} " + ", ".join(f"{p['id']} {p['box']}" for p in pan["panels"]) + (f"  [{pan['flag']}]" if pan.get("flag") else ""))
    return pan


def analyze_one_video(video: Path, cfg: dict, work: Path, sconf: dict, weights: Path, log, force=False, device=None) -> tuple[dict, dict, dict]:
    """-> ({view_id: TrackSet}, probe info, panels). Cached per video file hash + panel boxes."""
    vdir = work / video.stem; vdir.mkdir(parents=True, exist_ok=True)
    info = probe_video(video); dump_json(info, vdir / "probe.json")
    pan = panels_for(video, cfg, vdir, log)
    imgsz = int(cfg["params"].get("detector_imgsz", 640))
    key = {"sha256": info["sha256"], "boxes": [p["box"] for p in pan["panels"]], "weights": Path(weights).name, "width": cfg["params"]["track_max_width"], "imgsz": imgsz}
    cache = vdir / "tracks_cache.json"
    if cache.exists() and not force and json.loads(cache.read_text()) == key and all((vdir / f"tracks_{p['id']}.npz").exists() for p in pan["panels"]):
        views = {p["id"]: TrackSet.load(vdir / f"tracks_{p['id']}.npz") for p in pan["panels"]}
        log(f"    tracks cached: " + ", ".join(f"{k} {len(v.track_ids())} fragments" for k, v in views.items()))
        return views, info, pan
    rois = {}
    for p in pan["panels"]:
        if f"light_{p['id']}" in sconf:
            rois.setdefault(p["id"], {})["light"] = sconf[f"light_{p['id']}"]
    log(f"    tracking {info['width']}x{info['height']} {info['fps_nominal']:.2f} fps, {info['n_frames_nominal']} frames, {len(pan['panels'])} view(s) ...")
    views = analyze_video(video, weights, pan["panels"], max_width=cfg["params"]["track_max_width"], rois=rois, log=log, device=device, conf=cfg["params"]["pose_conf"], imgsz=imgsz)
    for vid, ts in views.items():
        split_fragments(ts); ts.meta["video_sha256"] = info["sha256"]; ts.save(vdir / f"tracks_{vid}.npz")
    cache.write_text(json.dumps(key))
    return views, info, pan


def flash_candidates(ts: TrackSet) -> list[dict]:
    fps = ts.meta.get("timing", {}).get("fps_measured") or ts.fps_nominal
    cands = []
    if "light" in ts.roi_series and len(ts.roi_series["light"]):
        cands += detect_flash_roi(ts.roi_series["light"], ts.frame_idx, ts.pts_ms, fps)
    for c in detect_flash_auto(ts.grid, ts.frame_idx, ts.pts_ms, fps):
        if not any(abs(c["onset_frame"] - r["onset_frame"]) <= 2 for r in cands):
            c["view"] = ts.view_id; cands.append(c)
    return sorted(cands, key=lambda c: c["onset_t_s"])


def plan_chunks(ts: TrackSet, subj: dict, offset_s: float, imu: dict, params: dict, fps_meas: float) -> list[dict]:
    dur = imu["duration_s"]; m = params["margin_s"]
    t = ts.pts_ms / 1000.0
    present = np.zeros(len(t), bool); present[subj["pos"]] = True
    lo, hi = max(offset_s - m, subj["pts_ms"][0] / 1000.0), min(offset_s + dur + m, subj["pts_ms"][-1] / 1000.0)
    pos = np.where((t >= lo) & (t <= hi))[0]
    step = 2 if fps_meas > params["fps_downsample_above"] else 1
    pos = pos[::step]
    if len(pos) < 30:
        return []
    # split at long absences (no subject for > 2 s) so the lifter never runs on frames without the subject
    segs, cur = [], [pos[0]]
    for a, b in zip(pos[:-1], pos[1:]):
        gap_no_subject = not present[a:b + 1].any()
        if (t[b] - t[cur[-1]] > 2.0 and not present[cur[-1]:b + 1].any()):
            if len(cur) >= 30:
                segs.append(np.array(cur))
            cur = [b]
        else:
            cur.append(b)
    if len(cur) >= 30:
        segs.append(np.array(cur))
    N = params["max_chunk_frames"]; ov = params["chunk_overlap_frames"]
    stand = [(a, b) for l, a, b in imu["events"] if l == "Standing"]
    chunks = []
    for seg in segs:
        # trim leading/trailing frames without the subject (keep <= 1 s margin)
        pres = present[seg]
        if not pres.any():
            continue
        first, last = int(np.argmax(pres)), len(seg) - 1 - int(np.argmax(pres[::-1]))
        seg = seg[max(0, first - int(fps_meas / step)):last + int(fps_meas / step) + 1]
        starts = [0]
        while starts[-1] + N < len(seg):
            ideal = starts[-1] + N - ov; best = ideal
            for i in range(starts[-1] + 2 * N // 3, ideal + 1):
                ti = t[seg[i]] - offset_s
                if any(a + 0.3 <= ti <= b - 0.3 for a, b in stand):
                    best = i
            starts.append(best)
        for s in starts:
            chunks.append({"chunk_id": len(chunks), "pos": seg[s:min(s + N, len(seg))], "frame_step": step})
    return chunks


def prepare_gvhmr_inputs(work: Path, cid: str, rid: str, video: Path, ts: TrackSet, subj: dict, offset_s: float, imu: dict, params: dict, fps_meas: float, log) -> dict:
    cdir = work / cid / rid / "chunks"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    chunks = plan_chunks(ts, subj, offset_s, imu, params, fps_meas)
    if not chunks:
        return {"n_chunks": 0, "reason": "subject not visible in this view inside the recording window"}
    bx0, by0, bx1, by1 = ts.box; pw = bx1 - bx0
    scale = min(1.0, params["gvhmr_max_width"] / pw)
    F = len(ts.frame_idx)
    box_full = np.full((F, 4), np.nan); box_full[subj["pos"]] = subj["box"]
    vis = np.zeros(F, bool); vis[subj["pos"]] = True
    idx = np.arange(F)
    for j in range(4):
        ok = np.isfinite(box_full[:, j]); box_full[:, j] = np.interp(idx, idx[ok], box_full[ok, j])
    box_local = (box_full - [bx0, by0, bx0, by0]) * scale
    out = []
    for c in chunks:
        pos = c["pos"]; frames = ts.frame_idx[pos]; name = f"chunk_{c['chunk_id']}"
        info = cut_video(video, cdir / f"{name}.mp4", frames, out_fps=30.0, scale=scale, crop=list(ts.box))
        if info["n_frames"] != len(frames):
            log(f"      WARNING {name}: cut has {info['n_frames']} frames, expected {len(frames)}")
        np.save(cdir / f"{name}_bbox.npy", box_local[pos].astype(np.float32))
        meta = {"chunk_id": c["chunk_id"], "video": f"{name}.mp4", "bbox": f"{name}_bbox.npy", "n_frames": int(len(frames)), "frame_idx": frames.tolist(),
                "pts_ms": ts.pts_ms[pos].tolist(), "imu_t_s": (ts.pts_ms[pos] / 1000.0 - offset_s).tolist(), "visible": vis[pos].tolist(),
                "frame_step": c["frame_step"], "cut_scale": scale, "crop": list(ts.box), "fps_source": fps_meas, "status": "pending", "cut": info,
                "camera_id": cid, "recording_id": rid}
        (cdir / f"{name}.json").write_text(json.dumps(meta))
        out.append({"chunk_id": c["chunk_id"], "n_frames": int(len(frames)), "t_video_s": [float(ts.pts_ms[pos[0]] / 1000), float(ts.pts_ms[pos[-1]] / 1000)]})
    return {"n_chunks": len(out), "chunks": out, "frame_step": chunks[0]["frame_step"], "cut_scale": scale, "crop": list(ts.box)}


def export_subject_track(subj: dict, out: Path, view_box: list, keep_face: bool = False):
    kp = subj["kp"].copy()
    if not keep_face:
        kp[:, :5, :] = np.nan
    np.savez_compressed(out, frame_idx=subj["frame_idx"], pts_ms=subj["pts_ms"], bbox_xyxy=subj["box"], det_conf=subj["conf"], kp2d=kp.astype(np.float32),
                        visibility=subj["vis"], src_track_id=subj["src_tid"], view_box=np.array(view_box),
                        keypoint_order="COCO-17 (x, y in full-frame px, conf); face points NaN unless --keep-face-keypoints; visibility 2 full / 1 partial")


# ----------------------------------------------------------------------------------------------------------------
def run_session_entry(entry: dict, root: Path, cfg: dict, weights: Path = DEFAULT_WEIGHTS, force: bool = False, keep_face: bool = False,
                      device: str | None = None, log=None, progress=None) -> dict:
    sid = entry["session_id"]; sdir = session_out(root, sid)
    work, deliver, cdir = sdir / "work", sdir / "deliver", sdir / "config"
    bundle = deliver / "validation_bundle"
    own_log = log is None
    log = log or tee_logger(bundle / "logs" / "run.log")
    params = cfg["params"]; sconf = _load_session_config(cdir); selp = {k: v for k, v in cfg["selection"].items() if not k.startswith("_")}
    t0 = time.time()
    summ = {"session_id": sid, "subject": entry["subject"], "dir": entry["dir"], "wgvp_version": __version__, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recordings": {}, "videos": {}, "items": {}, "status": "PASS", "flags": list(entry.get("flags", []))}
    log(f"[{sid}] {len(entry['recordings'])} recording(s), {len(entry['videos'])} video(s)")
    # ---- videos: panels + tracking (cached)
    vcache = {}
    for v in entry["videos"]:
        video = Path(v["path"])
        try:
            views, info, pan = analyze_one_video(video, cfg, work, sconf, Path(weights), log, force=force, device=device)
        except Exception as e:
            import traceback
            log(f"    [{video.name}] FAIL: {e}\n{traceback.format_exc()}")
            summ["videos"][v["video_id"]] = {"status": "FAIL", "error": str(e), "file": video.name}; summ["status"] = "FAIL"; continue
        flashes = {vid: flash_candidates(ts) for vid, ts in views.items()}
        timing = next(iter(views.values())).meta.get("timing", {}) if views else {}
        vrec = {"status": "PASS", "file": video.name, "video": {k: info[k] for k in ("width", "height", "fps_nominal", "n_frames_nominal", "codec_fourcc", "sha256", "size_bytes")},
                "timing": timing, "panels": {k: v for k, v in pan.items() if k != "preview_png"}, "views": {vid: {"n_fragments": int(len(ts.track_ids())), "n_detections": int(len(ts.det_tid)), "box": ts.box} for vid, ts in views.items()},
                "flash_candidates": {vid: f[:3] for vid, f in flashes.items()}, "flags": [], "tags": v.get("tags", "")}
        if timing.get("gap_frames_est", 0) > 0:
            vrec["flags"].append(f"{timing['gap_frames_est']} frame(s) missing by PTS ({timing['n_gaps']} gaps)")
        if timing.get("vfr"):
            vrec["flags"].append("variable frame rate (PTS used for timing)")
        if pan.get("flag"):
            vrec["flags"].append(pan["flag"])
        if len(pan["panels"]) == 1:
            vrec["flags"].append("single view detected (no second panel found)")
        summ["videos"][v["video_id"]] = vrec
        vcache[v["video_id"]] = {"views": views, "info": info, "pan": pan, "flashes": flashes, "path": video}
        for vid, ts in views.items():
            log(f"    [{video.name}/{vid}] {len(ts.track_ids())} fragments; flash: {[(c['onset_frame'], round(c['z'], 1)) for c in flashes[vid][:2]]}")
    # ---- label exports (video-time Walk intervals) per video, if any
    label_iv = {}
    for lp in entry.get("labels", []):
        try:
            iv = read_walk_intervals(Path(lp)); label_iv[Path(lp).name] = iv; log(f"    labels {Path(lp).name}: {len(iv)} Walk interval(s)")
        except Exception as e:
            log(f"    labels {Path(lp).name}: cannot read ({e})"); summ["flags"].append(f"label export unreadable: {Path(lp).name}: {e}")
    # ---- recordings
    cam_results = {}
    for ri, rec in enumerate(entry["recordings"]):
        rid = rec["recording_id"]
        if progress:
            progress("session", ri, len(entry["recordings"]), f"{sid}: {rid}")
        try:
            imu = load_recording(Path(rec["csv"]), work / "imu_ref", rid, rec["subject"])
        except Exception as e:
            log(f"  [{rid}] FAIL reading recording CSV: {e}"); summ["recordings"][rid] = {"status": "FAIL", "error": str(e)}; summ["status"] = "FAIL"; continue
        rs = summarize_imu_ref(imu); rs["status"] = "PASS"; rs["flags"] = []
        if rs["walk_s"] < 3:
            rs["flags"].append("under 3 s of annotated walking"); rs["status"] = "WARN"
        if rs["nan_fraction"] > 0.2:
            rs["flags"].append(f"pelvis accelerometer {rs['nan_fraction']:.0%} NaN"); rs["status"] = "WARN"
        summ["recordings"][rid] = rs
        log(f"  [{rid}] IMU {imu['duration_s']:.1f}s, walk {rs['walk_s']:.1f}s in {rs['n_walk_bouts']} bouts, walkway contact {rs['walkway_contact_fraction']:.2f}")
        cands = [v for v in rec["videos"] if v["video_id"] in vcache]
        if not cands:
            summ["items"][rid] = {"status": "FAIL", "flags": ["no analysable video for this recording"]}; summ["status"] = "FAIL"; continue
        # joint selection on every candidate video; keep the best pelvis match
        best = None
        for v in cands:
            vc = vcache[v["video_id"]]
            prim = joint_selection(vc["views"], imu, selp, log=log)
            log(f"    [{rid} vs {v['video_id']}] joint xcorr: ok={prim['ok']} conf={prim['confidence']} r={prim.get('r_fine', 0):.3f} margin={prim.get('margin', 0):.3f} offset={prim.get('offset_s')}")
            if best is None or (prim.get("r_fine") or 0) > (best[1].get("r_fine") or 0):
                best = (v, prim)
        v, prim = best
        vc = vcache[v["video_id"]]; views = vc["views"]; video = vc["path"]
        rec_item = {"video": video.name, "video_id": v["video_id"], "candidates": [c["video_id"] for c in cands], "status": "PASS", "flags": [], "views": {}}
        if len(cands) > 1:
            rec_item["flags"].append(f"chosen among {len(cands)} candidate videos by pelvis match")
        fps_meas = next(iter(views.values())).meta.get("timing", {}).get("fps_measured") or next(iter(views.values())).fps_nominal
        n_frames_video = int(vc["info"]["n_frames_nominal"]); video_len_s = n_frames_video / max(fps_meas, 1)
        # cross-checks: flash (any view), labels, sheet
        flash_all = sorted([c for f in vc["flashes"].values() for c in f], key=lambda c: c["onset_t_s"])
        lab = None
        for name, iv in label_iv.items():
            lab = walk_label_offset(iv, imu["events"], video_len_s, imu["duration_s"]); lab["file"] = name
            if lab.get("available"):
                break
        sheet = rec.get("sheet_start_frames", {}); sheet_frame = sheet.get("*") or next(iter(sheet.values()), None)
        ts0 = next(iter(views.values()))
        sync = combine_sync(prim, flash_all, lab, sheet_frame, ts0.frame_idx, ts0.pts_ms, fps_meas, tol_frames=params["tol_frames"])
        log(f"    [{rid}] sync {sync['status']} via {sync['source']} offset {None if sync['offset_s'] is None else round(sync['offset_s'], 3)} s (frame {sync['offset_frame']}) {sync['flags']}")
        d_rec = deliver / rid; d_rec.mkdir(parents=True, exist_ok=True)
        dump_json(sync, d_rec / "sync.json"); dump_json({**prim, "views": {vk: {kk: vv for kk, vv in vv_.items() if kk != "frame_masks"} for vk, vv_ in prim["views"].items()}}, d_rec / "selection_joint.json")
        rec_item.update(sync_status=sync["status"], sync_source=sync["source"], offset_s=sync["offset_s"], offset_frame=sync["offset_frame"], sync_flags=sync["flags"],
                        agreement=sync["agreement"], r_fine=prim.get("r_fine"), margin=prim.get("margin"), selection_confidence=prim.get("confidence"),
                        lag_ambiguity=prim.get("lag_ambiguity"), lag_candidates=prim.get("lag_candidates", [])[:3])
        if (prim.get("lag_ambiguity") or 0) > 0.85 and len(prim.get("lag_candidates", [])) > 1:
            alt = prim["lag_candidates"][1]
            rec_item["flags"].append(f"another person matches the IMU nearly as well at offset {alt['offset_s']:.2f} s (support ratio {prim['lag_ambiguity']:.2f}) - check the thumbnails")
        cam_results[rid] = {}
        bouts_cov = {}
        for vid, ts in views.items():
            key = f"{rid}/{vid}"; d_out = d_rec / vid; d_out.mkdir(parents=True, exist_ok=True)
            pv = prim["views"].get(vid)
            roi = roi_selection(ts, sconf.get(f"walkway_{vid}"), selp)
            pick = pick_selection(sconf.get(f"pick_{vid}_{rid}"), ts)
            dec = decide_view(pv, prim.get("confidence", "none"), roi, pick)
            item = {"status": _status_merge(dec["status"], sync["status"]), "flags": [], "selection_source": dec["source"], "subject_track_ids": dec["track_ids"],
                    "accepted_by_xcorr": (pv or {}).get("accepted_by_xcorr", []), "attached_by_appearance": (pv or {}).get("attached_by_appearance", []),
                    "r_fine": (pv or {}).get("r_fine"), "margin": (pv or {}).get("margin"), "view_confidence": (pv or {}).get("confidence"), "view_offset_s": (pv or {}).get("view_offset_s"),
                    "n_fragments": int(len(ts.track_ids())), "video": video.name, "view_box": ts.box, "selection_status": dec["status"], "sync_status": sync["status"],
                    "agreement": {"selection": dec["agreement"], "sync": sync["agreement"]}, "coverage": (pv or {}).get("coverage", {})}
            if dec["status"] != "PASS":
                item["flags"].append("selection:" + dec["message"])
            log(f"    [{key}] selection {dec['status']} via {dec['source']} tracks {dec['track_ids']} (r {item['r_fine']}, margin {item['margin']}); coverage present {item['coverage'].get('present_frac')}, full {item['coverage'].get('full_frac')}, gaps {item['coverage'].get('n_gaps')}")
            dump_json({k: v for k, v in dec.items() if k != "primary"} | {"primary": {kk: vv for kk, vv in (dec.get("primary") or {}).items() if kk != "frame_masks"}}, d_out / "selection.json")
            subj = None
            if dec["track_ids"]:
                masks = (pv or {}).get("frame_masks") if dec["source"].startswith("imu_xcorr") else None
                subj = merged_subject_track(ts, dec["track_ids"], masks)
                item["lag_consistency"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "window_r"} for k, v in ((pv or {}).get("consistency") or {}).items()}
                if masks:
                    item["flags"].append("part of a tracker fragment did not follow the subject's timing and was dropped (identity switch)")
                if len(subj["pos"]) == 0:
                    subj = None
                else:
                    export_subject_track(subj, d_out / "subject_track.npz", ts.box, keep_face); item["subject_frames"] = int(len(subj["pos"]))
            light = sconf.get(f"light_{vid}") or ({"cell": sync["flash"]["used"]["cell"]} if sync["flash"].get("used") and sync["flash"]["used"].get("view") == vid and "cell" in sync["flash"]["used"] else None)
            thumbs = write_thumbnails(ts, dec["track_ids"], bundle / "thumbnails", walkway=sconf.get(f"walkway_{vid}"), light=light, tag=f"{rid}__{vid}", offset_frame=sync["offset_frame"])
            item["thumbnails"] = [str(Path(t).relative_to(deliver)) for t in thumbs]
            if subj is not None and sync["offset_s"] is not None:
                prep = prepare_gvhmr_inputs(work, vid, rid, video, ts, subj, sync["offset_s"], imu, params, fps_meas, log)
                item["gvhmr_prep"] = prep
                if prep["n_chunks"] == 0:
                    item["flags"].append(prep["reason"]); item["status"] = _status_merge(item["status"], "WARN")
                log(f"    [{key}] lifter inputs: {prep['n_chunks']} chunk(s), step {prep.get('frame_step')}, scale {prep.get('cut_scale')}")
                t_hip, x_hip = hip_vertical_signal(subj)
                cam_results[rid][vid] = {"fps": fps_meas, "sync": sync, "hip": (t_hip - sync["offset_s"], x_hip), "pose": None, "view_offset_s": item["view_offset_s"]}
            else:
                cam_results[rid][vid] = {"fps": fps_meas, "sync": sync, "hip": None, "pose": None, "view_offset_s": item["view_offset_s"]}
            for b in item["coverage"].get("walk_bouts", []):
                bouts_cov.setdefault((b["t0"], b["t1"]), {})[vid] = b["covered_frac"]
            summ["items"][key] = item; rec_item["views"][vid] = item["status"]
            dump_json(item, d_out / "qc.json")
        # walk bouts: each must be covered (>= 80%) by at least one view
        bouts = [{"t0": a, "t1": b, "covered_by": {v: round(c, 2) for v, c in cov.items()}, "ok": any(c >= 0.8 for c in cov.values())} for (a, b), cov in sorted(bouts_cov.items())]
        rec_item["walk_bouts"] = bouts; rec_item["n_walk_bouts_ok"] = sum(1 for b in bouts if b["ok"]); rec_item["n_walk_bouts"] = len(bouts)
        if bouts and rec_item["n_walk_bouts_ok"] == 0:
            rec_item["flags"].append("no walk bout is covered by any view"); rec_item["status"] = "FAIL"
        elif bouts and rec_item["n_walk_bouts_ok"] < len(bouts):
            rec_item["flags"].append(f"{len(bouts) - rec_item['n_walk_bouts_ok']} of {len(bouts)} walk bouts not covered by any view"); rec_item["status"] = "WARN"
        rec_item["status"] = _status_merge(rec_item["status"], sync["status"], *rec_item["views"].values())
        summ["items"][rid] = rec_item
        dump_json(rec_item, d_rec / "qc.json")
        rc = reconcile_recording(cam_results[rid], tol_frames=params["tol_frames"])
        dump_json(rc, d_rec / "reconcile.json")
        summ["items"][f"{rid}/reconcile"] = {"status": rc["status"], "flags": rc["flags"], "pairs": _pairs_brief(rc)}
        log(f"  [{rid}] reconcile: {rc['status']} {rc['flags']}")
    summ["status"] = _status_merge(summ["status"], *[v["status"] for v in summ["items"].values() if v["status"] != "N/A"], *[v["status"] for v in summ["videos"].values()])
    summ["elapsed_s"] = time.time() - t0; summ["environment"] = environment_record()
    dump_json(summ, deliver / "session_summary.json"); dump_json(summ["environment"], bundle / "environment.json")
    # the panel previews are edge renders; copy them into the bundle for the QC summary
    for v in entry["videos"]:
        pp = work / Path(v["path"]).stem / "panels_preview.png"
        if pp.exists():
            (bundle / "panels").mkdir(parents=True, exist_ok=True); shutil.copy(pp, bundle / "panels" / f"{Path(v['path']).stem}_panels.png")
    pc = privacy_check(deliver); dump_json(pc, bundle / "privacy_check.json")
    summ["privacy_check"] = {"ok": pc["ok"], "n_files": pc["n_files"], "problems": pc["problems"]}
    dump_json(summ, deliver / "session_summary.json")
    log(f"[{sid}] {summ['status']} privacy_ok={pc['ok']} {summ['elapsed_s']:.0f}s")
    if own_log:
        log.close()
    return summ


def _pairs_brief(rec: dict) -> dict:
    out = {}
    for k, v in rec.get("pairs", {}).items():
        b = {kk: vv for kk, vv in v.items() if kk != "pose"}
        if v.get("pose"):
            b["pose"] = {kk: vv for kk, vv in v["pose"].items() if kk != "per_joint_geodesic"}
        out[k] = b
    return out


def refresh_session_after_gvhmr(root: Path, sid: str, cfg: dict, log=print) -> dict:
    sdir = session_out(root, sid); deliver = sdir / "deliver"; f = deliver / "session_summary.json"
    if not f.exists():
        return {}
    summ = json.loads(f.read_text()); cams = {}
    for key, item in list(summ["items"].items()):
        if "/" not in key or key.endswith("/reconcile"):
            continue
        rid, vid = key.split("/"); d = deliver / rid / vid
        sync = json.loads((deliver / rid / "sync.json").read_text()) if (deliver / rid / "sync.json").exists() else {}
        entry = {"fps": None, "sync": sync, "hip": None, "pose": None, "view_offset_s": item.get("view_offset_s")}
        st = d / "subject_track.npz"
        if st.exists() and sync.get("offset_s") is not None:
            z = np.load(st); tr = {"kp": z["kp2d"], "box": z["bbox_xyxy"], "pts_ms": z["pts_ms"]}
            t_hip, x_hip = hip_vertical_signal(tr); entry["hip"] = (t_hip - sync["offset_s"], x_hip)
        vrec = summ["videos"].get(summ["items"].get(rid, {}).get("video_id", ""), {})
        entry["fps"] = (vrec.get("timing") or {}).get("fps_measured") or (vrec.get("video") or {}).get("fps_nominal") or 30.0
        g = d / "smpl_global.npz"
        if g.exists():
            z = np.load(g); entry["pose"] = (z["imu_t_s"], z["body_pose"])
            item["gvhmr"] = {"n_frames": int(len(z["frame_idx"])), "finite": bool(np.isfinite(z["body_pose"]).all()), "visible_frames": int(z["visible"].sum())}
            if not item["gvhmr"]["finite"]:
                item["flags"].append("non-finite SMPL parameters"); item["status"] = "FAIL"
        else:
            chunks = sorted((sdir / "work" / vid / rid / "chunks").glob("chunk_*.json"))
            if chunks:
                sts = [json.loads(c.read_text()).get("status") for c in chunks]
                item["gvhmr"] = {"n_frames": 0, "chunk_status": sts}
                if any(s == "failed" for s in sts):
                    item["flags"].append("GVHMR failed on a chunk (see work/<view>/<recording>/gvhmr_run.log)"); item["status"] = "FAIL"
        cams.setdefault(rid, {})[vid] = entry
    for rid in cams:
        rc = reconcile_recording(cams[rid], tol_frames=cfg["params"]["tol_frames"])
        dump_json(rc, deliver / rid / "reconcile.json")
        summ["items"][f"{rid}/reconcile"] = {"status": rc["status"], "flags": rc["flags"], "pairs": _pairs_brief(rc)}
    summ["status"] = _status_merge(*[v["status"] for v in summ["items"].values() if v["status"] != "N/A"], *[v["status"] for v in summ["videos"].values()])
    pc = privacy_check(deliver); dump_json(pc, deliver / "validation_bundle" / "privacy_check.json")
    summ["privacy_check"] = {"ok": pc["ok"], "n_files": pc["n_files"], "problems": pc["problems"]}
    dump_json(summ, f)
    return summ


# ----------------------------------------------------------------------------------------------------------------
def run_gvhmr_stage(root: Path, cfg: dict, gvhmr_root: str | None = None, gvhmr_python: str | None = None, log=print, only: str | None = None) -> dict:
    groot = gvhmr_root or cfg["gvhmr"].get("root") or ""; gpy = gvhmr_python or cfg["gvhmr"].get("python") or sys.executable
    if not groot:
        return {"ran": False, "message": "no GVHMR root configured (config gvhmr.root or --gvhmr-root)"}
    cmd = [gpy, "-m", "wgvp.gvhmr_stage", "--root", str(root), "--gvhmr-root", groot, "--timeout", str(cfg["gvhmr"].get("timeout_s", 7200))]
    if only:
        cmd += ["--only", only]
    env = dict(os.environ, PYTHONPATH=str(PKG_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    logf = out_root(root) / "logs" / "gvhmr_stage.log"; logf.parent.mkdir(parents=True, exist_ok=True)
    log(f"GVHMR stage: {' '.join(cmd)}  (log: {logf})")
    t0 = time.time()
    with open(logf, "a") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        for line in p.stdout:
            lf.write(line); log("    " + line.rstrip())
        rc = p.wait()
    return {"ran": True, "returncode": rc, "elapsed_s": time.time() - t0, "cmd": " ".join(cmd), "log": str(logf)}


def run_root(root: Path, cfg: dict | None = None, gvhmr_root: str | None = None, gvhmr_python: str | None = None, only: list[str] | None = None,
             force: bool = False, keep_face: bool = False, device: str | None = None, cpu_only: bool = False, log=None, progress=None,
             weights: Path = DEFAULT_WEIGHTS) -> dict:
    root = Path(root).resolve(); cfg = cfg or load_config(root)
    oroot = out_root(root); oroot.mkdir(parents=True, exist_ok=True)
    log = log or tee_logger(oroot / "logs" / "run.log")
    log(f"wgvp {__version__} run_root {root}")
    man = scan_root(root, cfg); dump_json(man, oroot / "manifest.json")
    sessions = [s for s in man["sessions"] if not only or s["session_id"] in only]
    log(f"scan: {man['n_sessions']} sessions, {man['n_recordings']} recordings, {man['n_videos']} videos, {len(man['unmatched'])} unmatched; running {len(sessions)}")
    for i, s in enumerate(sessions):
        if progress:
            progress("root", i, len(sessions), s["session_id"])
        try:
            run_session_entry(s, root, cfg, weights=weights, force=force, keep_face=keep_face, device=device, log=log, progress=progress)
        except Exception as e:
            import traceback
            log(f"[{s['session_id']}] CRASH {e}\n{traceback.format_exc()}")
            dump_json({"session_id": s["session_id"], "status": "FAIL", "error": str(e), "items": {}, "videos": {}}, session_out(root, s["session_id"]) / "deliver" / "session_summary.json")
    gv = {"ran": False}
    if not cpu_only and (gvhmr_root or cfg["gvhmr"].get("root")):
        gv = run_gvhmr_stage(root, cfg, gvhmr_root, gvhmr_python, log=log)
        for s in sessions:
            refresh_session_after_gvhmr(root, s["session_id"], cfg, log=log)
    elif not cpu_only:
        gv["message"] = "GPU stage not run: set gvhmr.root / gvhmr.python in wgvp_config.json (or the GUI fields) and run `wgvp gvhmr <root>`"
        log(gv["message"])
    from .summary import build_summary
    summ = build_summary(root, cfg, gvhmr=gv, log=log)
    if hasattr(log, "close"):
        log.close()
    return summ
