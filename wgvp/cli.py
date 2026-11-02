"""Command line: `python -m wgvp <command>` (or `wgvp` after `pip install -e .`). No arguments launches the GUI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="wgvp", description=f"WearGait video-to-pose pipeline {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="open the GUI (default when no command is given)").add_argument("root", nargs="?")
    p = sub.add_parser("scan", help="find sessions / recordings / videos under a data folder"); p.add_argument("root")
    p = sub.add_parser("run", help="run every session under a data folder"); p.add_argument("root")
    p.add_argument("--gvhmr-root"); p.add_argument("--gvhmr-python"); p.add_argument("--only", nargs="*", help="session ids")
    p.add_argument("--cpu-only", action="store_true"); p.add_argument("--force", action="store_true", help="re-track cached videos")
    p.add_argument("--device", default=None, help="cpu / cuda for the 2D tracker"); p.add_argument("--keep-face-keypoints", action="store_true")
    p = sub.add_parser("gvhmr", help="run only the GPU stage on pending chunks (inside or via the GVHMR environment)"); p.add_argument("root")
    p.add_argument("--gvhmr-root"); p.add_argument("--gvhmr-python"); p.add_argument("--only")
    p = sub.add_parser("summary", help="rebuild the cross-session QC summary"); p.add_argument("root")
    p = sub.add_parser("dry-run", help="run the bundled sample and verify the known answers"); p.add_argument("--out"); p.add_argument("--gvhmr-root"); p.add_argument("--gvhmr-python"); p.add_argument("--cpu-only", action="store_true")
    p = sub.add_parser("check-gpu", help="what is missing for the GPU stage"); p.add_argument("--gvhmr-root"); p.add_argument("--gvhmr-python")
    p = sub.add_parser("init-config", help="write wgvp_config.json (matching rules) into the data folder"); p.add_argument("root"); p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("set-roi", help="draw the walkway polygon or the sync-light rectangle for a camera"); p.add_argument("root"); p.add_argument("--session", required=True); p.add_argument("--camera", required=True)
    p.add_argument("--kind", choices=["walkway", "light"], default="walkway"); p.add_argument("--frame", type=int, default=0); p.add_argument("--show", choices=["edges", "skeleton", "pixels"], default="edges")
    p.add_argument("--points", help="non-interactive: 'x,y;x,y;...' in original pixels")
    p = sub.add_parser("pick-subject", help="operator fallback: click the subject's skeleton (or give --track-id)"); p.add_argument("root"); p.add_argument("--session", required=True); p.add_argument("--recording", required=True); p.add_argument("--camera", required=True)
    p.add_argument("--frame", type=int); p.add_argument("--track-id", type=int); p.add_argument("--png", help="also write the skeleton frame to this PNG")
    p = sub.add_parser("make-sample", help="(analysis side) regenerate the bundled dry-run sample"); p.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd in (None, "gui"):
        from .gui import launch
        launch(getattr(a, "root", None)); return 0
    if a.cmd == "scan":
        from .config import load_config, write_default_config
        from .discovery import manifest_text, scan_root
        from .qc import dump_json
        root = Path(a.root); write_default_config(root); man = scan_root(root, load_config(root))
        (root / "wgvp_output").mkdir(exist_ok=True); dump_json(man, root / "wgvp_output" / "manifest.json")
        print(manifest_text(man)); return 0
    if a.cmd == "run":
        from .config import load_config, write_default_config
        from .session import run_root
        root = Path(a.root); write_default_config(root)
        s = run_root(root, load_config(root), gvhmr_root=a.gvhmr_root, gvhmr_python=a.gvhmr_python, only=a.only, force=a.force,
                     keep_face=a.keep_face_keypoints, device=a.device, cpu_only=a.cpu_only)
        print(json.dumps(s["totals"])); return 0 if s["totals"].get("FAIL", 0) == 0 else 1
    if a.cmd == "gvhmr":
        from .config import load_config
        from .session import run_gvhmr_stage, refresh_session_after_gvhmr
        from .summary import build_summary
        root = Path(a.root); cfg = load_config(root)
        r = run_gvhmr_stage(root, cfg, a.gvhmr_root, a.gvhmr_python, only=a.only)
        for d in sorted((root / "wgvp_output").glob("*/deliver/session_summary.json")):
            refresh_session_after_gvhmr(root, d.parent.parent.name, cfg)
        build_summary(root, cfg, gvhmr=r); return 0 if r.get("returncode", 1) == 0 else 1
    if a.cmd == "summary":
        from .config import load_config
        from .summary import build_summary
        build_summary(Path(a.root), load_config(Path(a.root))); return 0
    if a.cmd == "dry-run":
        from .dryrun import dry_run
        r = dry_run(a.out, a.gvhmr_root, a.gvhmr_python, a.cpu_only); return 0 if r["status"] == "PASS" else 1
    if a.cmd == "check-gpu":
        from .check_gpu import check, format_rows, run_in_interpreter
        if a.gvhmr_python:
            rc, out = run_in_interpreter(a.gvhmr_python, a.gvhmr_root); print(out); return rc
        rows = check(a.gvhmr_root); print(format_rows(rows)); return 0 if all(r[1] for r in rows) else 1
    if a.cmd == "init-config":
        from .config import write_default_config
        print(write_default_config(Path(a.root), overwrite=a.overwrite)); return 0
    if a.cmd in ("set-roi", "pick-subject"):
        from .config import load_config
        from .discovery import scan_root
        from .session import session_out
        from .tracking import TrackSet
        root = Path(a.root); man = scan_root(root, load_config(root))
        s = next((s for s in man["sessions"] if s["session_id"] == a.session), None)
        if s is None:
            print(f"session {a.session} not found; sessions: {[x['session_id'] for x in man['sessions']]}"); return 1
        cdir = session_out(root, a.session) / "config"
        tr = session_out(root, a.session) / "work" / a.camera / "tracks.npz"
        tracks = TrackSet.load(tr) if tr.exists() else None
        if a.cmd == "set-roi":
            from .roi_tool import set_roi
            vid = next((v["path"] for r in s["recordings"] for v in r["videos"] if v["camera_id"] == a.camera), None)
            if vid is None:
                print(f"camera {a.camera} not found in session {a.session}"); return 1
            pts = [tuple(float(x) for x in p.split(",")) for p in a.points.split(";")] if a.points else None
            print(set_roi(Path(vid), cdir, a.camera, a.kind, a.frame, a.show, pts, tracks)); return 0
        from .roi_tool import pick_subject
        if tracks is None:
            print(f"no tracks yet for {a.camera}: run the pipeline once (CPU) first"); return 1
        print(pick_subject(tracks, cdir, a.camera, a.recording, a.frame, a.track_id, Path(a.png) if a.png else None)); return 0
    if a.cmd == "make-sample":
        from tools.make_sample import main as mk
        return mk(a.out)
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
