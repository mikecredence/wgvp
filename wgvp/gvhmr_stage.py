"""GPU stage: GVHMR on the subject's cut video chunks, driven by OUR bbox track (so the lifter follows the subject,
not the longest track in the frame), then conversion to the pose-level deliverables.

Runs inside the GVHMR environment (conda env `gvhmr` on our box; the Docker image). Usage:
    python -m wgvp.gvhmr_stage --session <dir> --gvhmr-root /path/to/GVHMR [--timeout 7200] [--only rec/cam]

GVHMR's demo.py reuses <output_root>/<stem>/preprocess/bbx.pt when it exists, so we write it first (same layout as
its tracker: bbx_xyxy (L,4) and bbx_xys (L,3) via hmr4d's get_bbx_xys_from_xyxy). demo.py is called unchanged
with `-s` (static camera) and GVHMR_RENDER=0 (our render guard; if the guard is absent the QC videos are rendered
and deleted afterwards).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _log(msg):
    print(f"[gvhmr_stage] {msg}", flush=True)


def smooth_boxes(b: np.ndarray, win: int = 5, passes: int = 2) -> np.ndarray:
    out = b.astype(np.float64).copy()
    k = np.ones(win) / win
    for _ in range(passes):
        pad = np.pad(out, ((win // 2, win // 2), (0, 0)), mode="edge")
        out = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(out.shape[1])], 1)
    return out


def write_bbx_pt(gvhmr_root: Path, pred_root: Path, stem: str, bbox_xyxy: np.ndarray):
    sys.path.insert(0, str(gvhmr_root))
    import torch
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    bb = torch.tensor(np.asarray(bbox_xyxy, np.float32))
    xys = get_bbx_xys_from_xyxy(bb, base_enlarge=1.2).float()
    d = pred_root / stem / "preprocess"; d.mkdir(parents=True, exist_ok=True)
    torch.save({"bbx_xyxy": bb, "bbx_xys": xys}, d / "bbx.pt")


def run_demo(gvhmr_root: Path, video: Path, pred_root: Path, timeout: int, log_path: Path, f_mm: float | None = None) -> dict:
    demo = gvhmr_root / "tools/demo/demo.py"
    has_guard = "GVHMR_RENDER" in demo.read_text(encoding="utf-8")
    env = dict(os.environ, GVHMR_RENDER="0")
    cmd = [sys.executable, str(demo), "--video", str(video), "--output_root", str(pred_root), "-s"]
    if f_mm:                         # the view's focal length (35 mm equivalent, on the diagonal); demo.py takes whole millimetres
        cmd += ["--f_mm", str(int(round(float(f_mm))))]
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== {' '.join(cmd)}\n")
        try:
            p = subprocess.run(cmd, cwd=str(gvhmr_root), env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout,
                               stdin=subprocess.DEVNULL)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9; lf.write(f"\n[timeout after {timeout}s]\n")
    return {"returncode": rc, "elapsed_s": time.time() - t0, "render_guard": has_guard, "cmd": " ".join(cmd)}


def convert_results(pred_dir: Path) -> dict:
    """hmr4d_results.pt -> numpy dict (no features, no images)."""
    import torch
    blob = torch.load(pred_dir / "hmr4d_results.pt", map_location="cpu", weights_only=False)
    out = {}
    for fr in ("global", "incam"):
        sp = blob[f"smpl_params_{fr}"]
        out[fr] = {k: sp[k].detach().cpu().numpy().astype(np.float32) for k in ("global_orient", "body_pose", "betas", "transl")}
    out["K_fullimg"] = blob["K_fullimg"][0].detach().cpu().numpy().astype(np.float32)
    vp = pred_dir / "preprocess" / "vitpose.pt"
    out["kp2d_vitpose"] = torch.load(vp, map_location="cpu").detach().cpu().numpy().astype(np.float32) if vp.exists() else None
    bb = pred_dir / "preprocess" / "bbx.pt"
    if bb.exists():
        b = torch.load(bb, map_location="cpu"); out["bbx_xyxy"] = b["bbx_xyxy"].numpy().astype(np.float32)
    if "net_outputs" in blob and "static_conf_logits" in blob["net_outputs"]:
        out["static_conf_logits"] = blob["net_outputs"]["static_conf_logits"][0].detach().cpu().numpy().astype(np.float32)
    return out


def cleanup_pred_dir(pred_dir: Path):
    """Remove every pixel-bearing file GVHMR left (input copy, overlays, renders); keep the .pt results in work/."""
    for f in list(pred_dir.glob("*.mp4")) + list((pred_dir / "preprocess").glob("*.mp4")):
        f.unlink(missing_ok=True)


def process_chunk(gvhmr_root: Path, chunk_json: Path, timeout: int) -> dict:
    meta = json.loads(chunk_json.read_text(encoding="utf-8"))
    work = chunk_json.parent
    pred_root = work / "gvhmr"
    stem = Path(meta["video"]).stem
    video = work / meta["video"]
    bbox = np.load(work / meta["bbox"])
    res_pt = pred_root / stem / "hmr4d_results.pt"
    if not res_pt.exists():
        if (pred_root / stem).exists():
            shutil.rmtree(pred_root / stem)
        write_bbx_pt(gvhmr_root, pred_root, stem, smooth_boxes(bbox))
        r = run_demo(gvhmr_root, video.resolve(), pred_root.resolve(), timeout, work / "gvhmr_run.log", f_mm=meta.get("focal_35mm"))
        meta["gvhmr_run"] = r
        if not res_pt.exists():
            meta["status"] = "failed"; meta["error"] = f"GVHMR produced no hmr4d_results.pt (rc {r['returncode']}); see gvhmr_run.log"
            chunk_json.write_text(json.dumps(meta, indent=1), encoding="utf-8"); return meta
    conv = convert_results(pred_root / stem)
    n = len(conv["global"]["body_pose"])
    if n != meta["n_frames"]:
        meta["status"] = "failed"; meta["error"] = f"GVHMR returned {n} frames for a {meta['n_frames']}-frame chunk"
        chunk_json.write_text(json.dumps(meta, indent=1), encoding="utf-8"); return meta
    np.savez_compressed(work / f"{stem}_pose.npz",
                        **{f"global_{k}": v for k, v in conv["global"].items()},
                        **{f"incam_{k}": v for k, v in conv["incam"].items()},
                        K_fullimg=conv["K_fullimg"],
                        kp2d_vitpose=conv["kp2d_vitpose"] if conv["kp2d_vitpose"] is not None else np.zeros((n, 17, 3), np.float32),
                        bbx_xyxy=conv.get("bbx_xyxy", np.zeros((n, 4), np.float32)),
                        static_conf_logits=conv.get("static_conf_logits", np.zeros((n, 6), np.float32)))
    cleanup_pred_dir(pred_root / stem)
    meta["status"] = "done"; meta["pose_npz"] = f"{stem}_pose.npz"
    chunk_json.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def assemble(rec_cam_work: Path, deliver_dir: Path, keep_face_kp: bool = False) -> dict:
    """Merge chunk poses into deliver/<rec>/<cam>/smpl_{global,incam}.npz + kp2d_vitpose.npz + gvhmr_meta.json."""
    chunks = sorted(rec_cam_work.glob("chunk_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    metas = [json.loads(c.read_text(encoding="utf-8")) for c in chunks]
    if not metas or any(m.get("status") != "done" for m in metas):
        return {"ok": False, "reason": "not all chunks done", "chunks": [m.get("status") for m in metas]}
    missing = [m["pose_npz"] for m in metas if not (rec_cam_work / m["pose_npz"]).exists()]
    if missing:                      # chunk records say done but the GVHMR dumps are not here (a re-run from pulled-back records)
        return {"ok": False, "reason": "chunk pose file(s) missing", "missing": missing, "already_assembled": (deliver_dir / "smpl_global.npz").exists()}
    rows = {}   # source frame_idx -> (chunk_id, local index)
    for m in metas:
        fi = np.array(m["frame_idx"]); n = len(fi)
        for k in range(n):
            f = int(fi[k])
            if f in rows:
                # overlap: keep the chunk whose centre is closer to this frame
                c_old, k_old = rows[f]; n_old = len(metas[c_old]["frame_idx"])
                if abs(k - n / 2) >= abs(k_old - n_old / 2):
                    continue
            rows[f] = (m["chunk_id"], k)
    frames = np.array(sorted(rows)); src = [rows[f] for f in frames]
    poses = {m["chunk_id"]: np.load(rec_cam_work / m["pose_npz"]) for m in metas}
    def gather(key):
        return np.stack([poses[c][key][k] for c, k in src])
    meta0 = metas[0]
    pts = np.concatenate([np.array(m["pts_ms"]) for m in metas]); fidx = np.concatenate([np.array(m["frame_idx"]) for m in metas])
    imu_t = np.concatenate([np.array(m["imu_t_s"]) for m in metas]); vis = np.concatenate([np.array(m["visible"]) for m in metas])
    held = np.concatenate([np.array(m.get("rest_held") or [False] * len(m["visible"]), bool) for m in metas])
    lut = {int(f): i for i, f in enumerate(fidx)}
    sel = [lut[int(f)] for f in frames]
    deliver_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = np.array([c for c, _ in src], np.int16)
    # per-frame quality fields: the subject box touching an edge of the view (head or feet cut, half out of frame) and
    # the largest joint-rotation step from the previous lifted frame (NaN at a chunk start: a new GVHMR run begins there)
    boxes = {m["chunk_id"]: np.load(rec_cam_work / m["bbox"]) for m in metas}
    crop0 = meta0.get("crop") or [0, 0, 0, 0]; sc0 = float(meta0["cut_scale"]); Wc, Hc = (crop0[2] - crop0[0]) * sc0, (crop0[3] - crop0[1]) * sc0
    bl = np.stack([boxes[c][k] for c, k in src])
    edge = (bl[:, 0] <= 1.0) | (bl[:, 1] <= 1.0) | (bl[:, 2] >= Wc - 1.0) | (bl[:, 3] >= Hc - 1.0)
    bp = gather("global_body_pose").reshape(len(frames), -1, 3)
    step = np.full(len(frames), np.nan, np.float32)
    if len(frames) > 1:
        from scipy.spatial.transform import Rotation as R
        d = (R.from_rotvec(bp[:-1].reshape(-1, 3)).inv() * R.from_rotvec(bp[1:].reshape(-1, 3))).magnitude().reshape(len(frames) - 1, -1)
        step[1:] = np.degrees(d.max(1)); step[1:][chunk_ids[1:] != chunk_ids[:-1]] = np.nan
    common = dict(frame_idx=frames.astype(np.int64), pts_ms=pts[sel].astype(np.float64), imu_t_s=imu_t[sel].astype(np.float64),
                  visible=vis[sel].astype(bool), rest_held=held[sel].astype(bool), chunk_id=chunk_ids, edge_truncated=edge.astype(bool), pose_step_deg=step,
                  fps_source=float(meta0["fps_source"]), gvhmr_fps_assumed=30.0, frame_step=int(meta0["frame_step"]),
                  fields_doc="visible: the subject was detected in this frame (False = no detection: the box is interpolated over a short gap, or held at rest); "
                             "rest_held: no detection, but the IMU labels the body at rest and the box of the neighbouring detection is kept (the merge uses such a frame "
                             "when its keypoints agree with that detection's, see wgvp.fusion.seen_frames; any other frame without a detection is not to be trusted); edge_truncated: the subject box touches a view edge; "
                             "pose_step_deg: largest per-joint rotation step from the previous frame (NaN at a chunk start); chunk_id: separate GVHMR runs, split at long absences")
    for fr in ("global", "incam"):
        np.savez_compressed(deliver_dir / f"smpl_{fr}.npz", global_orient=gather(f"{fr}_global_orient"), body_pose=gather(f"{fr}_body_pose"),
                            betas=gather(f"{fr}_betas"), transl=gather(f"{fr}_transl"), frame=fr,
                            note=("GVHMR gravity-aligned world frame (y up)" if fr == "global" else "camera frame"), **common)
    # ViTPose keypoints / boxes are in the cut (cropped, scaled) frame: map back to full-frame pixels of the original video
    sc = float(meta0["cut_scale"]); crop = meta0.get("crop") or [0, 0, 0, 0]; off = np.array([crop[0], crop[1]], np.float32)
    kp = gather("kp2d_vitpose").astype(np.float32); kp[:, :, :2] = kp[:, :, :2] / sc + off
    if not keep_face_kp:
        kp[:, :5, :] = np.nan
    bb = gather("bbx_xyxy").astype(np.float32) / sc + np.array([crop[0], crop[1], crop[0], crop[1]], np.float32)
    np.savez_compressed(deliver_dir / "kp2d_vitpose.npz", kp2d=kp, bbox_xyxy=bb, static_conf_logits=gather("static_conf_logits"),
                        K_fullimg_cut=poses[meta0["chunk_id"]]["K_fullimg"], cut_scale=sc, view_box=np.array(crop),
                        keypoint_order="COCO-17 (x, y full-frame px, conf); face points NaN unless --keep-face-keypoints; K_fullimg_cut is for the cut frame", **common)
    jumps = frames[np.nan_to_num(step, nan=0.0) > 30.0]
    gm = {"n_frames": int(len(frames)), "n_visible": int(vis[sel].sum()), "n_bridged": int((~vis[sel] & ~held[sel]).sum()), "n_rest_held": int(held[sel].sum()), "n_edge_truncated": int(edge.sum()),
          "n_pose_jumps_over_30deg": int(len(jumps)), "pose_jump_frames": jumps[:200].tolist(), "pose_step_deg_p99": (None if not np.isfinite(step).any() else float(np.nanpercentile(step, 99))),
          "chunks": [{k: m[k] for k in ("chunk_id", "n_frames", "status") if k in m} | {"gvhmr_run": m.get("gvhmr_run")} for m in metas],
          "frame_step": int(meta0["frame_step"]), "cut_scale": float(meta0["cut_scale"]), "view_box": meta0.get("crop"), "fps_source": float(meta0["fps_source"]),
          "gvhmr": {"static_camera": True, "checkpoint": "gvhmr_siga24_release.ckpt", "render": False, "focal_35mm": meta0.get("focal_35mm")},
          "files": ["smpl_global.npz", "smpl_incam.npz", "kp2d_vitpose.npz"]}
    (deliver_dir / "gvhmr_meta.json").write_text(json.dumps(gm, indent=1), encoding="utf-8")
    return {"ok": True, "n_frames": int(len(frames)), "n_chunks": len(metas)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="data root (all sessions under <root>/wgvp_output)")
    ap.add_argument("--session", default=None, help="one session output dir (<root>/wgvp_output/<session_id>)")
    ap.add_argument("--gvhmr-root", required=True)
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per chunk")
    ap.add_argument("--only", default=None, help="restrict to <recording>/<camera>")
    ap.add_argument("--keep-face-keypoints", action="store_true")
    a = ap.parse_args(argv)
    if not a.root and not a.session:
        _log("FAIL: give --root or --session"); return 2
    scan_dirs = [Path(a.session)] if a.session else sorted(p for p in (Path(a.root) / "wgvp_output").glob("*") if (p / "work").exists())
    root = Path(a.gvhmr_root)
    if not (root / "tools/demo/demo.py").exists():
        _log(f"FAIL: {root} is not a GVHMR checkout (tools/demo/demo.py missing)"); return 2
    for ck in ("inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt", "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
               "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt", "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"):
        if not (root / ck).exists():
            _log(f"FAIL: missing GVHMR weight {ck}"); return 2
    try:
        import torch
        if not torch.cuda.is_available():
            _log("FAIL: torch.cuda.is_available() is False in this environment"); return 2
        _log(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        _log(f"FAIL: torch not importable here ({e}); run this stage inside the GVHMR environment"); return 2
    jobs = [j for sd in scan_dirs for j in sorted(sd.glob("work/*/*/chunks/chunk_*.json"))]
    if a.only:
        parts = a.only.split("/")
        if len(parts) == 3:
            jobs = [j for j in jobs if j.parts[-6] == parts[0] and j.parts[-4] == parts[2] and j.parts[-3] == parts[1]]
        else:
            rec, cam = parts[-2], parts[-1]
            jobs = [j for j in jobs if j.parts[-4] == cam and j.parts[-3] == rec]
    _log(f"{len(jobs)} chunk(s) in {len(scan_dirs)} session dir(s)")
    failed = 0
    done_dirs = set()
    for j in jobs:
        meta = json.loads(j.read_text(encoding="utf-8"))
        if meta.get("status") == "done":
            _log(f"skip {j.parent.parent.name}/{j.stem} (done)"); done_dirs.add(j.parent); continue
        _log(f"run  {j.parts[-4]}/{j.parts[-3]}/{j.stem} ({meta['n_frames']} frames)")
        m = process_chunk(root, j, a.timeout)
        _log(f"     -> {m['status']} {m.get('error', '')} ({m.get('gvhmr_run', {}).get('elapsed_s', 0):.0f}s)")
        failed += m["status"] != "done"; done_dirs.add(j.parent)
    for d in sorted(done_dirs):
        cam, rec = d.parts[-3], d.parts[-2]; session = d.parents[3]; out = session / "deliver" / rec / cam
        try:
            r = assemble(d, out, keep_face_kp=a.keep_face_keypoints)
        except Exception as e:       # one view's assembly must not take the others down
            r = {"ok": False, "reason": f"assembly failed: {e!r}"}; failed += 1
        if not r.get("ok") and r.get("already_assembled"):
            _log(f"assemble {session.name}/{rec}/{cam}: kept the existing output (chunk dumps not on this machine)"); continue
        if not r.get("ok"):
            failed += 1
        _log(f"assemble {session.name}/{rec}/{cam}: {r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
