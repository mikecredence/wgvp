"""Command line: `python -m wgvp <command>` (or `wgvp` after `pip install -e .`). No arguments launches the GUI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def merged_sessions(root: Path) -> list[str]:
    """The sessions under <root>/wgvp_output that hold a merged recording (deliver/<recording>/merged_smpl.npz)."""
    return sorted({d.parents[2].name for d in (Path(root) / "wgvp_output").glob("*/deliver/*/merged_smpl.npz")})


def _q(path) -> str:
    """A path as it can be pasted into a command line: quoted when it holds a space."""
    return f'"{path}"' if " " in str(path) else str(path)


def no_tkinter_message() -> str:
    """What to do when this Python has no tkinter: the fix per platform, and the same steps on the command line."""
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    return "\n".join([
        "The GUI needs tkinter, and this Python was built without it.",
        f"  macOS, Homebrew Python:  brew install python-tk@{v}",
        "  Debian / Ubuntu:         sudo apt install python3-tk",
        "  python.org installers:   tkinter is included; on Windows re-run the installer and tick 'tcl/tk and IDLE'",
        "Every GUI button has a command, so nothing is blocked:",
        "  python -m wgvp dry-run --cpu-only            # Dry run (bundled sample)",
        "  python -m wgvp scan <data folder>            # Scan",
        "  python -m wgvp run  <data folder> --cpu-only # Run all sessions",
        "The QC summary is then <data folder>/wgvp_output/QC_SUMMARY.html.",
    ])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for stream in (sys.stdout, sys.stderr):                   # a folder name the console cannot show must not stop a run
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(prog="wgvp", description=f"WearGait video-to-pose pipeline {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="open the GUI (default when no command is given)").add_argument("root", nargs="?")
    p = sub.add_parser("scan", help="find sessions / recordings / videos under a data folder"); p.add_argument("root")
    p = sub.add_parser("run", help="run every session under a data folder"); p.add_argument("root")
    p.add_argument("--gvhmr-root"); p.add_argument("--gvhmr-python"); p.add_argument("--only", nargs="*", help="session ids")
    p.add_argument("--cpu-only", action="store_true"); p.add_argument("--force", action="store_true", help="re-track cached videos")
    p.add_argument("--device", default=None, help="cpu / cuda for the 2D tracker"); p.add_argument("--keep-face-keypoints", action="store_true")
    p.add_argument("--lift-device", choices=["cuda", "cpu"], help="cpu: run the lift (GVHMR) without a GPU; the same result, slowly")
    p = sub.add_parser("gvhmr", help="run only the GPU stage on pending chunks (inside or via the GVHMR environment)"); p.add_argument("root")
    p.add_argument("--gvhmr-root"); p.add_argument("--gvhmr-python"); p.add_argument("--only")
    p.add_argument("--lift-device", choices=["cuda", "cpu"], help="cpu: run the lift (GVHMR) without a GPU; the same result, slowly")
    p = sub.add_parser("summary", help="rebuild the cross-session QC summary"); p.add_argument("root")
    p = sub.add_parser("demo", help="render the demo video of a lifted recording from the pipeline's outputs (demo extras; the file holds video frames and stays under work/)")
    p.add_argument("root"); p.add_argument("--session"); p.add_argument("--recording"); p.add_argument("--gt", help="a rendered sample's <sid>_gt.npz: adds the truth panels and the error")
    p.add_argument("--out"); p.add_argument("--body-models"); p.add_argument("--t0", type=float); p.add_argument("--t1", type=float); p.add_argument("--step", type=int, default=1)
    p.add_argument("--gif"); p.add_argument("--gif-span", help="A,B seconds of IMU time for the GIF (default: the first 14 s of the span)")
    p.add_argument("--readout", choices=["both", "charts", "strips"], default="both"); p.add_argument("--strip-step", type=float, default=2.0)
    p.add_argument("--target", choices=["auto", "merge", "jointfit"], default="auto",
                   help="the pose shown: the recording's training target as target.json names it (default), or the merge / the joint two-view fit by name (the file name then gets _merge / _jointfit)")
    p = sub.add_parser("merge", help="(re)build the training target from existing GVHMR outputs: the merge (merged_smpl.npz), the joint two-view fit (jointfit_smpl.npz) and target.json; no GPU needed")
    p.add_argument("root"); p.add_argument("--gvhmr-root", help="the GVHMR checkout, for its SMPL-X model file (default: gvhmr.root in the config)"); p.add_argument("--body-models", help="folder that holds smplx/SMPLX_NEUTRAL.npz")
    p = sub.add_parser("jointfit", help="fit one body to both views' keypoints at once (jointfit_smpl.npz beside merged_smpl.npz) and settle target.json; two-view recordings, CPU, needs the SMPL-X model file. "
                                        "The pipeline runs it after every merge; use this to run it alone")
    p.add_argument("root"); p.add_argument("--session"); p.add_argument("--recording"); p.add_argument("--body-models", help="folder that holds smplx/SMPLX_NEUTRAL.npz (default: jointfit.body_models in the config, then the GVHMR checkout's)")
    p = sub.add_parser("dry-run", help="run the bundled sample and verify the known answers"); p.add_argument("--out"); p.add_argument("--gvhmr-root"); p.add_argument("--gvhmr-python"); p.add_argument("--cpu-only", action="store_true")
    p = sub.add_parser("check-gpu", help="what is missing for the GPU stage"); p.add_argument("--gvhmr-root"); p.add_argument("--gvhmr-python")
    p = sub.add_parser("init-config", help="write wgvp_config.json (matching rules) into the data folder"); p.add_argument("root"); p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("set-roi", help="draw the walkway polygon or the sync-light rectangle for a camera"); p.add_argument("root"); p.add_argument("--session", required=True); p.add_argument("--camera", required=True)
    p.add_argument("--kind", choices=["walkway", "light"], default="walkway"); p.add_argument("--frame", type=int, default=0); p.add_argument("--show", choices=["edges", "skeleton", "pixels"], default="edges")
    p.add_argument("--points", help="non-interactive: 'x,y;x,y;...' in original pixels")
    p = sub.add_parser("pick-subject", help="operator fallback: click the subject's skeleton (or give --track-id)"); p.add_argument("root"); p.add_argument("--session", required=True); p.add_argument("--recording", required=True); p.add_argument("--camera", required=True)
    p.add_argument("--frame", type=int); p.add_argument("--track-id", type=int); p.add_argument("--png", help="also write the skeleton frame to this PNG")
    p.add_argument("--pick", nargs="+", metavar="FRAME:X,Y", help="headless, machine-independent pick: one or more points on the subject (full-frame pixels), e.g. 986:303,89 1371:323,85")
    p = sub.add_parser("make-sample", help="(analysis side) regenerate the bundled dry-run sample"); p.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd in (None, "gui"):
        try:
            import tkinter  # noqa: F401
        except ImportError:
            print(no_tkinter_message()); return 1
        from .gui import launch
        launch(getattr(a, "root", None)); return 0
    if getattr(a, "root", None) is not None and not Path(a.root).is_dir():
        print(f"data folder not found: {a.root}"); return 2
    if a.cmd == "scan":
        from .config import load_config, write_default_config
        from .discovery import manifest_text, scan_root
        from .qc import dump_json
        root = Path(a.root); write_default_config(root); man = scan_root(root, load_config(root))
        (root / "wgvp_output").mkdir(exist_ok=True); dump_json(man, root / "wgvp_output" / "manifest.json")
        print(manifest_text(man))
        if not man["sessions"]:
            print("nothing to process here: the data folder should hold one folder per subject, with that subject's recording CSVs "
                  "and videos inside (README, 'Expected input').")
            print(f"matching rules: {root / 'wgvp_config.json'} (edit and scan again if your names differ)")
        return 0
    if a.cmd == "run":
        from .config import load_config, write_default_config
        from .session import run_root
        root = Path(a.root); write_default_config(root); cfg = load_config(root)
        if a.lift_device:
            cfg["gvhmr"]["device"] = a.lift_device
        s = run_root(root, cfg, gvhmr_root=a.gvhmr_root, gvhmr_python=a.gvhmr_python, only=a.only, force=a.force,
                     keep_face=a.keep_face_keypoints, device=a.device, cpu_only=a.cpu_only)
        if a.cpu_only:
            print("CPU stages done (subject chosen, sync found, lifter inputs cut). The poses come from the GPU stage: "
                  f"`python -m wgvp gvhmr {_q(root)}` on the machine with GVHMR (README, Quick start, step 5).")
        print(json.dumps(s["totals"])); return 0 if s["totals"].get("FAIL", 0) == 0 else 1
    if a.cmd == "gvhmr":
        from .config import load_config
        from .session import run_gvhmr_stage, refresh_session_after_gvhmr
        from .summary import build_summary
        root = Path(a.root); cfg = load_config(root)
        if a.lift_device:
            cfg["gvhmr"]["device"] = a.lift_device
        r = run_gvhmr_stage(root, cfg, a.gvhmr_root, a.gvhmr_python, only=a.only)
        if not r.get("ran"):
            print(f"GPU stage not run: {r.get('message', 'see the log')}"); return 1
        for d in sorted((root / "wgvp_output").glob("*/deliver/session_summary.json")):
            refresh_session_after_gvhmr(root, d.parent.parent.name, cfg, gvhmr_root=a.gvhmr_root)
        build_summary(root, cfg, gvhmr=r); return 0 if r.get("returncode", 1) == 0 else 1
    if a.cmd == "demo":
        from .demo import render_demo
        root = Path(a.root); sessions = [a.session] if a.session else merged_sessions(root)
        if not sessions:
            print("no merged recording under wgvp_output (run the GPU stage, then `wgvp merge`)"); return 1
        for sid in sessions:
            render_demo(root, sid, a.recording, a.gt, a.out if len(sessions) == 1 else None, a.body_models, a.t0, a.t1, a.step, a.gif if len(sessions) == 1 else None, a.gif_span, a.readout, a.strip_step, target=a.target)
        return 0
    if a.cmd == "merge":
        from .config import load_config
        from .session import refresh_session_after_gvhmr
        from .summary import build_summary
        root = Path(a.root); cfg = load_config(root); n = bad = nfit = 0
        for d in sorted((root / "wgvp_output").glob("*/deliver/session_summary.json")):
            s = refresh_session_after_gvhmr(root, d.parent.parent.name, cfg, gvhmr_root=a.gvhmr_root, body_models=a.body_models)
            st = [v.get("status") for k, v in (s or {}).get("items", {}).items() if k.endswith("/merge")]
            n += len(st); bad += sum(1 for x in st if x == "FAIL" or x is None)
            nfit += sum(1 for k, v in (s or {}).get("items", {}).items() if k.endswith("/target") and v.get("method") == "jointfit")
        build_summary(root, cfg)
        print(f"{n - bad} recording(s) merged, the joint two-view fit is the training target of {nfit} (target.json beside each says which file and why)"
              + (f", {bad} FAILED or without an export (a merged_smpl.npz from an earlier run, if there is one, was left as it was)" if bad else ""))
        return 0 if n and not bad else 1
    if a.cmd == "jointfit":
        from .config import load_config
        from .jointfit import jointfit_root
        from .session import refresh_session_targets
        from .summary import build_summary
        root = Path(a.root)
        try:
            out = jointfit_root(root, a.body_models, a.session, a.recording)
        except FileNotFoundError as e:
            print(e); return 1
        for d in sorted((root / "wgvp_output").glob("*/deliver/session_summary.json")):
            refresh_session_targets(root, d.parent.parent.name)
        if out:
            build_summary(root, load_config(root))
        n = sum(1 for s in out if s["status"] != "SKIP"); print(f"{n} recording(s) fitted, {len(out) - n} skipped"); return 0 if n else 1
    if a.cmd == "summary":
        from .config import load_config
        from .summary import build_summary
        if not (Path(a.root) / "wgvp_output").is_dir():
            print(f"nothing has been run under {a.root} yet (no wgvp_output folder): start with `python -m wgvp run {_q(a.root)} --cpu-only`"); return 1
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
        pts = [(int(q.split(":")[0]), float(q.split(":")[1].split(",")[0]), float(q.split(":")[1].split(",")[1])) for q in (a.pick or [])]
        print(pick_subject(tracks, cdir, a.camera, a.recording, a.frame, a.track_id, Path(a.png) if a.png else None, points=pts or None)); return 0
    if a.cmd == "make-sample":
        from tools.make_sample import main as mk
        return mk(a.out)
    ap.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())
