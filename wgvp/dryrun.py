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


def _r(x, n=3):
    """A measured number as it should read in a check line: three decimals, anything else untouched."""
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    return round(float(x), n) if isinstance(x, float) else x


def _check(name, ok, detail, checks, log):
    checks.append({"check": name, "ok": bool(ok), "detail": detail}); log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _check_merged_target(out: Path, sid: str, rid: str, ss: dict, exp: dict, checks: list, log) -> None:
    """The merged training target of the dry-run recording against the rendered truth (GPU runs)."""
    mg = ss["items"].get(f"{rid}/merge", {}); mf = session_out(out, sid) / "deliver" / rid / "merged_smpl.npz"
    if mf.exists():
        from .fusion import LEGS, fit_yaw, geodesic, step_stats
        m = np.load(mf); gt = np.load(SAMPLE_DIR / exp["gt_npz"]); gi = {int(f): i for i, f in enumerate(gt["frame_idx"])}
        keep = m["export_ok"] & np.array([int(f) in gi for f in m["frame_idx"]])
        sel = [gi[int(f)] for f in m["frame_idx"][keep]]
        body = m["body_pose"][keep].reshape(-1, 21, 3); gt_body = gt["pose_aa"][sel, 3:66].reshape(-1, 21, 3)
        legs = float(np.nanmedian(geodesic(body[:, LEGS], gt_body[:, LEGS]))) if keep.any() else float("nan")
        ex = (mg.get("export") or {}).get("frac_of_lifted_frames")
        _check("merged target written; its export keeps at least 70 % of the lifted frames", ex is not None and ex >= 0.7 and bool(np.isfinite(m["betas"]).all()),
               f"export {ex} of {mg.get('n_frames')} frames in {(mg.get('export') or {}).get('n_runs')} run(s); reference view {mg.get('reference_view')}; status {mg.get('status')} {mg.get('flags')}", checks, log)
        _check("merged legs within 15 deg of the rendered truth (median over exported frames)", np.isfinite(legs) and legs < 15.0,
               f"legs median {legs:.1f} deg over {int(keep.sum())} exported frames (each view alone against the truth: tools/two_view_fusion.py)", checks, log)
        root = m["global_orient"][keep]; gt_root = gt["pose_aa"][sel, :3]; okr = np.isfinite(root).all(1)
        yaw = fit_yaw(root, gt_root, okr) if okr.sum() > 10 else 0.0
        from scipy.spatial.transform import Rotation as Rot
        root_al = np.full_like(root, np.nan); root_al[okr] = (Rot.from_euler("y", yaw, degrees=True) * Rot.from_rotvec(root[okr])).as_rotvec()
        r_err = float(np.nanmedian(geodesic(root_al, gt_root))) if okr.any() else float("nan"); r_step = step_stats(m["global_orient"][:, None, :])["p99_deg"]
        _check("merged root continuous and within 15 deg of the truth (after one constant heading)", np.isfinite(r_err) and r_err < 15.0 and r_step < 12.0,
               f"root median error {r_err:.1f} deg (heading offset {yaw:+.0f} deg), frame step p99 {r_step:.1f} deg", checks, log)
        hr = m["global_orient_heading_rel"]; okh = np.isfinite(hr).all(1)
        fwd = Rot.from_rotvec(hr[okh]).apply([0.0, 0.0, 1.0]) if okh.any() else np.zeros((0, 3))
        _check("heading-relative root has no yaw left in it", okh.any() and float(np.abs(np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]))).max()) < 1.0,
               f"max residual heading {float(np.abs(np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]))).max()) if okh.any() else float('nan'):.2f} deg over {int(okh.sum())} frames", checks, log)
    else:
        _check("merged target exists", False, f"no merged_smpl.npz ({mg})", checks, log)


def _check_target(out: Path, sid: str, rid: str, exp: dict, checks: list, log) -> None:
    """target.json names the training target; where that is the joint two-view fit, the fit against the rendered
    truth (GPU runs). Without the SMPL-X model file the fit does not run and the merge is the target: that passes,
    and the detail says why."""
    rec = session_out(out, sid) / "deliver" / rid; tf = rec / "target.json"
    if not tf.exists():
        _check("training target named", False, "no target.json", checks, log); return
    t = json.loads(tf.read_text()); f = rec / str(t.get("target"))
    _check("training target named, and the file is there", t.get("target") is not None and f.exists(), f"{t.get('target')} ({t.get('why')})", checks, log)
    if t.get("method") != "jointfit" or not f.exists():
        return
    from .fusion import LEGS, geodesic
    m = np.load(f); gt = np.load(SAMPLE_DIR / exp["gt_npz"]); gi = {int(x): i for i, x in enumerate(gt["frame_idx"])}
    keep = m["export_ok"] & np.array([int(x) in gi for x in m["frame_idx"]]); sel = [gi[int(x)] for x in m["frame_idx"][keep]]
    body = m["body_pose"][keep].reshape(-1, 21, 3); gt_body = gt["pose_aa"][sel, 3:66].reshape(-1, 21, 3)
    legs = float(np.nanmedian(geodesic(body[:, LEGS], gt_body[:, LEGS]))) if keep.any() else float("nan")
    mg = np.load(rec / "merged_smpl.npz"); same = bool(np.array_equal(m["frame_idx"], mg["frame_idx"]) and np.array_equal(m["export_ok"], mg["export_ok"]))
    _check("joint fit (the target) on the merge's frames, legs within 15 deg of the rendered truth", same and np.isfinite(legs) and legs < 15.0,
           f"legs median {legs:.1f} deg over {int(keep.sum())} exported frames; fitted body height {t.get('body_height_m')} m", checks, log)


def dry_run(out_dir: str | None = None, gvhmr_root: str | None = None, gvhmr_python: str | None = None, cpu_only: bool = False, log=print, keep: bool = True) -> dict:
    exp = json.loads((SAMPLE_DIR / "expected.json").read_text())
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="wgvp_dryrun_"))
    if out.exists() and any(out.iterdir()):
        shutil.rmtree(out)
    shutil.copytree(SAMPLE_DIR / "data", out, dirs_exist_ok=True)      # the default folder comes from mkdtemp and already exists
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
           f"estimated {_r(off)} s vs true {exp['offset_s']} s via {ri.get('sync_source')} (r {_r(ri.get('r_fine'))}, margin {_r(ri.get('margin'))})", checks, log)
    fx = (ri.get("agreement") or {}).get("flash_minus_xcorr_s")
    _check("optional light cross-check agrees", fx is not None and abs(fx) <= tol, f"flash - xcorr = {_r(fx)} s", checks, log)
    for vid, e in exp["views"].items():
        it = ss["items"].get(f"{rid}/{vid}", {}); d = session_out(out, sid) / "deliver" / rid / vid
        st = d / "subject_track.npz"
        if st.exists():
            z = np.load(st); gt = np.array(e["gt_hip_px"]); gf = np.array(e["gt_frames"]); lut = {int(f): i for i, f in enumerate(gf)}
            errs, inside, wrong = [], [], 0
            boxes = z["bbox_xyxy"] if "bbox_xyxy" in z.files else np.full((len(z["frame_idx"]), 4), np.nan)
            for f, kp, bb in zip(z["frame_idx"], z["kp2d"], boxes):
                if int(f) in lut and np.isfinite(kp[11:13, :2]).all():
                    g = gt[lut[int(f)]]; errs.append(np.hypot(*(kp[11:13, :2].mean(0) - g)))
                    w, h = bb[2] - bb[0], bb[3] - bb[1]                          # identity: the truth hip lies in the chosen box (10 % slack)
                    inside.append(bool(np.isfinite(bb).all() and bb[0] - 0.1 * w <= g[0] <= bb[2] + 0.1 * w and bb[1] - 0.1 * h <= g[1] <= bb[3] + 0.1 * h))
                elif int(f) not in lut and np.isfinite(kp[11:13, :2]).all():
                    wrong += 1
            errs = np.array(errs); frac = float(np.mean(inside)) if inside else 0.0; near = float((errs < 0.06 * 600).mean()) if len(errs) else 0.0
            _check(f"{vid}: chosen track is the rendered subject", frac >= 0.8 and len(errs) > 30 and wrong < 0.1 * max(len(errs), 1),
                   f"{frac:.0%} of {len(errs)} frames hold the GT hip inside the chosen box ({near:.0%} within 36 px of the detector's hip, median {np.median(errs) if len(errs) else float('nan'):.1f} px); "
                   f"{wrong} frames claimed while the subject is out of this view; via {it.get('selection_source')}, r {_r(it.get('r_fine'))}, margin {_r(it.get('margin'))}, fragments {it.get('subject_track_ids')}", checks, log)
        else:
            _check(f"{vid}: subject track exists", False, f"no subject_track.npz (selection {it.get('selection_status')}: {it.get('flags')})", checks, log)
        cov = it.get("coverage", {})
        exp_present = e["n_visible"] / exp["n_frames_in_recording"]
        # the selection is judged on what the tracker produced: of the frames in which some fragment's box holds the
        # subject's hip, the chosen track must hold nearly all (the frames in which no fragment does are the detector's
        # limit, reported beside it: the walker standing half out of the room-wide side view's edge, for instance)
        gt = np.array(e["gt_hip_px"]); gf = np.array(e["gt_frames"]); lut = {int(f): i for i, f in enumerate(gf)}
        tracks_npz = session_out(out, sid) / "work" / Path(vrec.get("file", "")).stem / f"tracks_{vid}.npz"
        detectable, chosen = set(), set()
        if tracks_npz.exists():
            from .tracking import TrackSet
            ts = TrackSet.load(tracks_npz); frames = ts.frame_idx[ts.det_frame]
            for f, bb in zip(frames, ts.det_box):
                if int(f) in lut:
                    g = gt[lut[int(f)]]; w, h = bb[2] - bb[0], bb[3] - bb[1]
                    if bb[0] - 0.1 * w <= g[0] <= bb[2] + 0.1 * w and bb[1] - 0.1 * h <= g[1] <= bb[3] + 0.1 * h:
                        detectable.add(int(f))
        if st.exists():
            chosen = {int(f) for f in np.load(st)["frame_idx"] if int(f) in lut}
        recall = len(chosen & detectable) / max(len(detectable), 1)
        _check(f"{vid}: the chosen track holds the subject wherever the tracker found it", tracks_npz.exists() and len(detectable) > 30 and recall >= 0.9,
               f"{recall:.0%} of the {len(detectable)} frames in which some fragment's box holds the subject's hip are in the chosen track (the tracker found the subject on "
               f"{len(detectable) / max(e['n_visible'], 1):.0%} of the {e['n_visible']} frames it is in this view); present {cov.get('present_frac')} (scene {exp_present:.2f}), gaps {cov.get('n_gaps')} {cov.get('gaps_imu_time_s', [])[:4]}", checks, log)
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
           f"view offsets {_r(pair.get('view_offsets_s'))} differ by {_r(vd)} s; 2D pelvis local agreement r {_r(hip.get('r_best'))} at lag {_r(hip.get('residual_lag_s'))} s", checks, log)
    if gpu_ran and pair.get("pose"):
        _check("two views agree on SMPL leg joints", pair["pose"].get("legs_median_deg", 99) < 25 and abs(pair["pose"]["knee_residual_lag"].get("residual_lag_s", 9)) <= cfg["params"]["tol_frames"] / 30.0,
               f"legs median {pair['pose'].get('legs_median_deg')} deg, knee residual lag {pair['pose']['knee_residual_lag'].get('residual_lag_s')} s", checks, log)
    if gpu_ran:
        _check_merged_target(out, sid, rid, ss, exp, checks, log); _check_target(out, sid, rid, exp, checks, log)
    _check("walk bouts covered by at least one view", ri.get("n_walk_bouts", 0) > 0 and ri.get("n_walk_bouts_ok") == ri.get("n_walk_bouts"), f"{ri.get('n_walk_bouts_ok')}/{ri.get('n_walk_bouts')}", checks, log)
    _check("privacy check on the deliverables", ss.get("privacy_check", {}).get("ok"), f"{ss.get('privacy_check')}", checks, log)
    ok = all(c["ok"] for c in checks)
    res = {"status": "PASS" if ok else "FAIL", "checks": checks, "elapsed_s": time.time() - t0, "out": str(out), "gpu_stage_ran": gpu_ran, "html": str(out / "wgvp_output" / "QC_SUMMARY.html")}
    (out / "wgvp_output" / "dry_run_result.json").write_text(json.dumps(res, indent=1))
    log(f"DRY RUN {res['status']} ({sum(c['ok'] for c in checks)}/{len(checks)} checks, {res['elapsed_s']:.0f}s) -> {out}/wgvp_output")
    return res
