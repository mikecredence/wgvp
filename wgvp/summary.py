"""One-file QC summary across all sessions (QC_SUMMARY.html + qc_summary.csv), built from the per-session
session_summary.json files. Self-contained HTML; the only images are the skeleton-only thumbnails (sparse PNGs)."""
from __future__ import annotations

import base64
import csv
import html
import json
import time
from pathlib import Path

from . import __version__
from .qc import dump_json, is_skeleton_only

COLS = ["session", "recording", "view", "status", "video", "n_fragments", "selection_source", "subject_tracks", "r_fine", "margin",
        "present_frac", "full_frac", "partial_frac", "n_gaps", "lag_ambiguity", "sync_source", "offset_s", "offset_frame", "flash_minus_xcorr_s", "labels_minus_xcorr_s",
        "sheet_minus_xcorr_s", "walk_bouts_ok", "gvhmr_frames", "reconcile_status", "residual_lag_s", "legs_median_deg", "merge_status", "export_frac", "flags"]


def _fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return str(v)


def collect_rows(root: Path, out_root: Path) -> tuple[list[dict], dict]:
    rows, totals = [], {"PASS": 0, "WARN": 0, "FAIL": 0, "sessions": 0}
    for f in sorted(out_root.glob("*/deliver/session_summary.json")):
        s = json.loads(f.read_text()); sid = s["session_id"]; totals["sessions"] += 1
        totals[s.get("status", "FAIL")] = totals.get(s.get("status", "FAIL"), 0) + 1
        items = s.get("items", {})
        recs, merges = {}, {}
        for key, it in items.items():
            if key.endswith("/reconcile"):
                recs[key.split("/")[0]] = it
            elif key.endswith("/merge"):
                merges[key.split("/")[0]] = it
        for key, it in items.items():
            if key.endswith("/reconcile") or key.endswith("/merge") or "/" not in key:
                continue
            rid, vid = key.split("/")
            rc = recs.get(rid, {}); ri = items.get(rid, {}); mg = merges.get(rid, {})
            pair = next(iter(rc.get("pairs", {}).values()), {}) if rc else {}
            hip = pair.get("hip_vertical", {}) or {}; pose = pair.get("pose", {}) or {}
            ag = ri.get("agreement", {}) or {}; cov = it.get("coverage", {}) or {}
            rows.append({"session": sid, "recording": rid, "view": vid, "status": it.get("status"), "video": it.get("video", ""),
                         "n_fragments": it.get("n_fragments"), "selection_source": it.get("selection_source"), "subject_tracks": it.get("subject_track_ids"),
                         "r_fine": it.get("r_fine"), "margin": it.get("margin"), "present_frac": cov.get("present_frac"), "full_frac": cov.get("full_frac"),
                         "partial_frac": cov.get("partial_frac"), "n_gaps": cov.get("n_gaps"), "lag_ambiguity": ri.get("lag_ambiguity"), "sync_source": ri.get("sync_source"), "offset_s": ri.get("offset_s"),
                         "offset_frame": ri.get("offset_frame"), "flash_minus_xcorr_s": ag.get("flash_minus_xcorr_s"), "labels_minus_xcorr_s": ag.get("labels_minus_xcorr_s"),
                         "sheet_minus_xcorr_s": ag.get("sheet_minus_xcorr_s"), "walk_bouts_ok": f"{ri.get('n_walk_bouts_ok')}/{ri.get('n_walk_bouts')}",
                         "gvhmr_frames": (it.get("gvhmr") or {}).get("n_frames"), "reconcile_status": rc.get("status"),
                         "residual_lag_s": (pose.get("knee_residual_lag") or {}).get("residual_lag_s", pair.get("view_offset_diff_s")),
                         "legs_median_deg": pose.get("legs_median_deg"), "merge_status": mg.get("status"), "export_frac": (mg.get("export") or {}).get("frac_of_lifted_frames"),
                         "flags": (it.get("flags", []) + ri.get("flags", []) + ri.get("sync_flags", []) + mg.get("flags", [])),
                         "thumbnails": it.get("thumbnails", []), "_deliver": str(f.parent)})
    return rows, totals


def build_summary(root: Path, cfg: dict, gvhmr: dict | None = None, log=print, thumbnails: bool = True) -> dict:
    root = Path(root); out_root = root / "wgvp_output"
    man = json.loads((out_root / "manifest.json").read_text()) if (out_root / "manifest.json").exists() else {"unmatched": [], "sessions": []}
    rows, totals = collect_rows(root, out_root)
    with open(out_root / "qc_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in COLS})
    env = {}
    for f in out_root.glob("*/deliver/validation_bundle/environment.json"):
        env = json.loads(f.read_text()); break
    css = ("body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:18px;color:#222;max-width:1500px}"
           "table{border-collapse:collapse;font-size:12.5px}th,td{border:1px solid #ddd;padding:3px 6px;vertical-align:top}"
           "th{background:#f3f3f3;position:sticky;top:0}.PASS{background:#e6f6e6}.WARN{background:#fff4d6}.FAIL{background:#ffe0e0}"
           ".k{color:#666;font-size:12px}img{height:110px;border:1px solid #999;margin:2px;background:#000}h2{margin-top:26px}"
           "code{background:#f3f3f3;padding:1px 4px}")
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>wgvp QC summary</title><style>{css}</style></head><body>",
         f"<h1>WearGait video pipeline - QC summary</h1><p class='k'>wgvp {__version__} - {time.strftime('%Y-%m-%d %H:%M')} - root {html.escape(str(root))}</p>",
         f"<p><b>Sessions:</b> {totals['sessions']} &nbsp; <span class='PASS'>PASS {totals.get('PASS', 0)}</span> &nbsp; <span class='WARN'>WARN {totals.get('WARN', 0)}</span> &nbsp; <span class='FAIL'>FAIL {totals.get('FAIL', 0)}</span>"
         f" &nbsp; <b>recording-camera items:</b> {len(rows)} &nbsp; <b>unmatched files:</b> {len(man.get('unmatched', []))}</p>",
         "<p class='k'>What to look at: every row is one recording in one VIEW of its combined video. <b>selection_source</b> should read imu_xcorr with r_fine &gt; 0.35 and a margin &gt; 0.12 over the next person; "
         "<b>present/full/partial_frac</b> say how much of the recording the subject was in that view (fully / partly in frame); <b>n_gaps</b> counts exits; <b>walk_bouts_ok</b> = annotated walk bouts covered by at least one view; "
         "<b>sync_source</b> is the pelvis cross-correlation, '(validated by ...)' lists the cross-checks that agree (light flash, Walk-label export, annotation sheet); "
         "<b>residual_lag_s</b> is how far apart the two views place IMU t=0 (after GVHMR: the knee-angle residual lag; should be ~0); <b>merge_status</b> / <b>export_frac</b>: the merged training target (merged_smpl.npz) and the share of lifted frames its export keeps. Thumbnails are skeleton-only renders (green = chosen subject, grey = other people) at the IMU start frame; the panel previews are edge maps with the detected view boxes.</p>"]
    if gvhmr:
        h.append(f"<p class='k'>GPU stage: {html.escape(json.dumps({k: v for k, v in gvhmr.items() if k != 'cmd'}))}</p>")
    if env:
        h.append("<p class='k'>Environment: " + html.escape(", ".join(f"{k}={v}" for k, v in env.items())) + "</p>")
    h.append("<h2>Items</h2><table><tr>" + "".join(f"<th>{c}</th>" for c in COLS) + ("<th>skeleton at IMU t=0</th>" if thumbnails else "") + "</tr>")
    for r in rows:
        cells = "".join(f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in COLS)
        img = ""
        if thumbnails:
            # the thumbnail nearest the sync frame: the one whose name contains the offset frame
            pick = None
            for t in r.get("thumbnails", []):
                if r.get("offset_frame") is not None and f"f{int(r['offset_frame']):07d}" in t:
                    pick = t
            pick = pick or (r.get("thumbnails") or [None])[len(r.get("thumbnails", [])) // 2]
            if pick:
                p = Path(r["_deliver"]) / pick
                if p.exists() and is_skeleton_only(p):
                    img = f"<img src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}' title='{html.escape(pick)}'>"
        h.append(f"<tr class='{r.get('status')}'>{cells}" + (f"<td>{img}</td>" if thumbnails else "") + "</tr>")
    h.append("</table>")
    h.append("<h2>Sessions, recordings and videos found</h2><table><tr><th>session</th><th>folder</th><th>recordings (candidate videos)</th><th>label export</th><th>sheet</th><th>panels (edge preview)</th></tr>")
    for s in man.get("sessions", []):
        recs = "<br>".join(html.escape(f"{r['recording_id']}: " + (", ".join(Path(v['path']).name for v in r['videos']) or 'NO VIDEO') + (f"  [{r['flag']}]" if r.get('flag') else "")) for r in s["recordings"])
        pv = ""
        for v in s.get("videos", []):
            p = out_root / s["session_id"] / "deliver" / "validation_bundle" / "panels" / f"{Path(v['path']).stem}_panels.png"
            if thumbnails and p.exists() and is_skeleton_only(p, max_nonblack_frac=0.5):
                pv += f"<div class='k'>{html.escape(Path(v['path']).name)}</div><img style='height:150px' src='data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}'>"
        h.append(f"<tr><td>{html.escape(s['session_id'])}</td><td>{html.escape(s['dir'])}</td><td>{recs}</td><td>{', '.join(html.escape(Path(x).name) for x in s.get('labels', [])) or '-'}</td><td>{html.escape(Path(s['sheet']).name) if s.get('sheet') else '-'}</td><td>{pv}</td></tr>")
    h.append("</table>")
    if man.get("unmatched"):
        h.append(f"<h2>Unmatched files ({len(man['unmatched'])})</h2><ul>" + "".join(f"<li><code>{html.escape(u['path'])}</code> - {html.escape(u['reason'])}</li>" for u in man["unmatched"][:500]) + "</ul>")
    h.append("<h2>Privacy</h2><p class='k'>deliver/ folders contain NPZ/JSON/CSV/log files and skeleton-only PNGs; no video frame is written to any deliverable (privacy_check.json in each validation bundle lists every file).</p>")
    h.append("</body></html>")
    (out_root / "QC_SUMMARY.html").write_text("\n".join(h), encoding="utf-8")
    summary = {"totals": totals, "n_items": len(rows), "n_unmatched": len(man.get("unmatched", [])), "html": str(out_root / "QC_SUMMARY.html"),
               "csv": str(out_root / "qc_summary.csv"), "gvhmr": gvhmr}
    dump_json(summary, out_root / "qc_summary.json")
    log(f"QC summary: {summary['totals']} -> {summary['html']}")
    return summary
