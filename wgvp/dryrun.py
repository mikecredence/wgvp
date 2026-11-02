"""Dry run on the bundled sample: one combined two-view video (unequal panels, letterboxed side view) rendered from
a public WearGait recording with a co-walker, a seated operator close to the side camera and a bystander. The
pipeline must find the panels, pick the rendered subject in both views, recover the known sync offset, record the
subject's exits from the side view as gaps, and (with a GVHMR environment) lift both views with agreeing poses."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .session import run_root, session_out

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample"


def _check(name, ok, detail, checks, log):
    checks.append({"check": name, "ok": bool(ok), "detail": detail}); log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def dry_run(out_dir: str | None = None, gvhmr_root: str | None = None, gvhmr_python: str | None = None, cpu_only: bool = False, log=print, keep: bool = True) -> dict:
    exp = json.loads((SAMPLE_DIR / "expected.json").read_text())
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="wgvp_dryrun_"))
    if out.exists() and any(out.iterdir()):
        shutil.rmtree(out)
    shutil.copytree(SAMPLE_DIR / "data", out)
    log(f"dry run: sample copied to {out}")
    t0 = time.time(); cfg = load_config(out)
    summ = run_root(out, cfg, gvhmr_root=gvhmr_root, gvhmr_python=gvhmr_python, cpu_only=cpu_only, log=log)
    checks = []
    sid, rid = exp["session_id"], exp["recording_id"]
    ss = json.loads((session_out(out, sid) / "deliver" / "session_summary.json").read_text())
    gpu_ran = bool(summ.get("gvhmr", {}).get("ran")) and summ["gvhmr"].get("returncode") == 0
    vrec = next(iter(ss["videos"].values()), {})
    pan = {p["id"]: p["box"] for p in (vrec.get("panels", {}).get("panels") or [])}
    tolpx = 24
    _check("two panels detected at the right boxes", vrec.get("panels", {}).get("layout") == exp["layout"] and all(
        k in pan and max(abs(a - b) for a, b in zip(pan[k], exp["panels"][k])) <= tolpx for k in exp["panels"]), f"detected {vrec.get('panels', {}).get('layout')} {pan} vs expected {exp['panels']}", checks, log)
    ri = ss["items"].get(rid, {})
    off = ri.get("offset_s"); tol = 2.0 / exp["fps"]
    _check("sync offset within 2 frames of truth (pelvis cross-correlation)", off is not None and abs(off - exp["offset_s"]) <= tol and "pelvis" in str(ri.get("sync_source")),
           f"estimated {off} s vs true {exp['offset_s']} s via {ri.get('sync_source')} (r {ri.get('r_fine')}, margin {ri.get('margin')})", checks, log)
    fx = (ri.get("agreement") or {}).get("flash_minus_xcorr_s")
    _check("optional light cross-check agrees", fx is not None and abs(fx) <= tol, f"flash - xcorr = {fx} s", checks, log)
    for vid, e in exp["views"].items():
        it = ss["items"].get(f"{rid}/{vid}", {}); d = session_out(out, sid) / "deliver" / rid / vid
        st = d / "subject_track.npz"
        if st.exists():
            z = np.load(st); gt = np.array(e["gt_hip_px"]); gf = np.array(e["gt_frames"]); lut = {int(f): i for i, f in enumerate(gf)}
            errs, wrong = [], 0
            for f, kp in zip(z["frame_idx"], z["kp2d"]):
                if int(f) in lut and np.isfinite(kp[11:13, :2]).all():
                    errs.append(np.hypot(*(kp[11:13, :2].mean(0) - gt[lut[int(f)]])))
                elif int(f) not in lut and np.isfinite(kp[11:13, :2]).all():
                    wrong += 1
            errs = np.array(errs); frac = float((errs < 0.06 * 600).mean()) if len(errs) else 0.0
            _check(f"{vid}: chosen track is the rendered subject", frac >= 0.8 and len(errs) > 30 and wrong < 0.1 * max(len(errs), 1),
                   f"{frac:.0%} of {len(errs)} frames within 36 px of the GT hip (median {np.median(errs) if len(errs) else float('nan'):.1f} px); {wrong} frames claimed while the subject is out of this view; "
                   f"via {it.get('selection_source')}, r {it.get('r_fine')}, margin {it.get('margin')}, fragments {it.get('subject_track_ids')}", checks, log)
        else:
            _check(f"{vid}: subject track exists", False, f"no subject_track.npz (selection {it.get('selection_status')}: {it.get('flags')})", checks, log)
        cov = it.get("coverage", {})
        exp_present = e["n_visible"] / exp["n_frames_in_recording"]
        _check(f"{vid}: coverage reported and consistent with the scene", cov.get("present_frac") is not None and abs(cov["present_frac"] - exp_present) < 0.15,
               f"present {cov.get('present_frac')} (scene: {exp_present:.2f}), full {cov.get('full_frac')}, partial {cov.get('partial_frac')}, gaps {cov.get('n_gaps')} {cov.get('gaps_imu_time_s', [])[:4]}", checks, log)
        if vid == "right":
            _check("right view: the subject's exits are recorded as gaps (no bystander attached)", cov.get("n_gaps", 0) >= 1 and cov.get("present_frac", 1) < 0.9, f"{cov.get('n_gaps')} gaps, present {cov.get('present_frac')}", checks, log)
        if gpu_ran:
            g = d / "smpl_global.npz"
            if g.exists():
                z = np.load(g)
                _check(f"{vid}: GVHMR output complete and finite", bool(np.isfinite(z["body_pose"]).all()) and len(z["frame_idx"]) >= 0.8 * e["n_visible"],
                       f"{len(z['frame_idx'])} frames (subject visible in {e['n_visible']})", checks, log)
            else:
                _check(f"{vid}: GVHMR output exists", False, f"no smpl_global.npz ({it.get('gvhmr')})", checks, log)
    rc = ss["items"].get(f"{rid}/reconcile", {}); pair = next(iter(rc.get("pairs", {}).values()), {}); hip = pair.get("hip_vertical", {}) or {}
    vd = pair.get("view_offset_diff_s")
    _check("two views agree on the IMU timeline (each view's own placement of IMU t=0)", vd is not None and abs(vd) <= 2.0 / 30,
           f"view offsets {pair.get('view_offsets_s')} differ by {vd} s; 2D pelvis local agreement r {hip.get('r_best')} at lag {hip.get('residual_lag_s')} s", checks, log)
    if gpu_ran and pair.get("pose"):
        _check("two views agree on SMPL leg joints", pair["pose"].get("legs_median_deg", 99) < 25 and abs(pair["pose"]["knee_residual_lag"].get("residual_lag_s", 9)) <= cfg["params"]["tol_frames"] / 30.0,
               f"legs median {pair['pose'].get('legs_median_deg')} deg, knee residual lag {pair['pose']['knee_residual_lag'].get('residual_lag_s')} s", checks, log)
    _check("walk bouts covered by at least one view", ri.get("n_walk_bouts", 0) > 0 and ri.get("n_walk_bouts_ok") == ri.get("n_walk_bouts"), f"{ri.get('n_walk_bouts_ok')}/{ri.get('n_walk_bouts')}", checks, log)
    _check("privacy check on the deliverables", ss.get("privacy_check", {}).get("ok"), f"{ss.get('privacy_check')}", checks, log)
    ok = all(c["ok"] for c in checks)
    res = {"status": "PASS" if ok else "FAIL", "checks": checks, "elapsed_s": time.time() - t0, "out": str(out), "gpu_stage_ran": gpu_ran, "html": str(out / "wgvp_output" / "QC_SUMMARY.html")}
    (out / "wgvp_output" / "dry_run_result.json").write_text(json.dumps(res, indent=1))
    log(f"DRY RUN {res['status']} ({sum(c['ok'] for c in checks)}/{len(checks)} checks, {res['elapsed_s']:.0f}s) -> {out}/wgvp_output")
    return res
