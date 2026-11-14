"""Minimal tkinter GUI: pick the data folder, scan, run, stop, watch progress, open the QC summary, render the demo video. No extra installs."""
from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__
from .config import config_path, load_config, write_default_config
from .discovery import manifest_text, scan_root
from .progress import RunProgress, RunStopped

BAR = 1000       # the progress bar's full length, in its own units


# the demo child's progress lines (read for the bar) and macOS's own chatter about the video libraries it loads
_NOISE = re.compile(r"^\[demo\] |ApplePersistenceIgnoreState|^objc\[\d+\]: Class |^mysterious crashes\. One of the duplicates")


def _noise(line: str) -> bool:
    return bool(_NOISE.search(line))


class NotStarted(Exception):
    """A job that cannot begin; the message goes to the user in a dialog, not as a traceback in the log."""


def launch(initial_root: str | None = None, on_ready=None):
    """Open the window. on_ready(app), if given, is called once before the event loop starts (the tests drive the window through it)."""
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox

    app = tk.Tk(); app.title(f"WearGait video pipeline (wgvp {__version__})"); app.geometry("1160x760")
    q: queue.Queue = queue.Queue()
    state = {"manifest": None, "running": False, "summary": None, "rp": RunProgress(), "prefix": "", "stop": threading.Event(), "job": ""}

    def log(msg):
        q.put(("log", str(msg)))

    def progress(stage, i, n, msg):
        q.put(("progress", (stage, i, n, msg)))

    # --- top: folders
    top = ttk.Frame(app, padding=8); top.pack(fill="x")
    ttk.Label(top, text="Data folder (top level):").grid(row=0, column=0, sticky="w")
    v_root = tk.StringVar(value=initial_root or ""); e_root = ttk.Entry(top, textvariable=v_root, width=80); e_root.grid(row=0, column=1, sticky="we", padx=4)
    ttk.Button(top, text="Browse...", command=lambda: v_root.set(filedialog.askdirectory() or v_root.get())).grid(row=0, column=2)
    ttk.Button(top, text="Scan", command=lambda: do_scan()).grid(row=0, column=3, padx=4)
    ttk.Label(top, text="GVHMR folder (GPU stage, optional):").grid(row=1, column=0, sticky="w")
    v_groot = tk.StringVar(); ttk.Entry(top, textvariable=v_groot, width=80).grid(row=1, column=1, sticky="we", padx=4)
    ttk.Button(top, text="Browse...", command=lambda: v_groot.set(filedialog.askdirectory() or v_groot.get())).grid(row=1, column=2)
    ttk.Label(top, text="Python of the GVHMR environment:").grid(row=2, column=0, sticky="w")
    v_gpy = tk.StringVar(); ttk.Entry(top, textvariable=v_gpy, width=80).grid(row=2, column=1, sticky="we", padx=4)
    ttk.Button(top, text="Browse...", command=lambda: v_gpy.set(filedialog.askopenfilename() or v_gpy.get())).grid(row=2, column=2)
    ttk.Button(top, text="Check GPU", command=lambda: do_check_gpu()).grid(row=2, column=3, padx=4)
    top.columnconfigure(1, weight=1)

    # --- middle: found items
    mid = ttk.Frame(app, padding=(8, 0)); mid.pack(fill="both", expand=True)
    tree = ttk.Treeview(mid, columns=("info", "status"), height=12); tree.heading("#0", text="session / recording / camera"); tree.heading("info", text="file"); tree.heading("status", text="status")
    tree.column("#0", width=330); tree.column("info", width=430); tree.column("status", width=250)
    tree.pack(fill="both", expand=True)
    previews = {}
    def _open_preview(e):
        it = tree.focus()
        if it in previews and previews[it]:
            webbrowser.open(Path(previews[it]).as_uri())
    tree.bind("<Double-1>", _open_preview)
    ttk.Label(mid, text="Scan shows every subject folder, recording CSV and video found, and the detected view panels (double-click a 'panels' row to see the boxes). Fix a wrong guess in wgvp_config.json (matching rules / views.override) and re-scan.", wraplength=1050).pack(anchor="w")
    # --- buttons
    bar = ttk.Frame(app, padding=8); bar.pack(fill="x")
    b_run = ttk.Button(bar, text="Run all sessions", command=lambda: do_run()); b_run.pack(side="left")
    b_dry = ttk.Button(bar, text="Dry run (sample)", command=lambda: do_dry()); b_dry.pack(side="left", padx=6)
    b_stop = ttk.Button(bar, text="Stop", command=lambda: do_stop(), state="disabled"); b_stop.pack(side="left", padx=6)
    v_cpu = tk.BooleanVar(value=False); ttk.Checkbutton(bar, text="CPU stages only (skip GVHMR)", variable=v_cpu).pack(side="left", padx=6)
    ttk.Button(bar, text="Open QC summary", command=lambda: do_open()).pack(side="right")
    ttk.Button(bar, text="Output folder", command=lambda: do_open_folder()).pack(side="right", padx=6)
    b_demo = ttk.Button(bar, text="Demo video", command=lambda: do_demo()); b_demo.pack(side="right", padx=6)
    pb = ttk.Progressbar(app, mode="determinate", maximum=BAR); pb.pack(fill="x", padx=8)
    v_prog = tk.StringVar(value="idle"); ttk.Label(app, textvariable=v_prog).pack(anchor="w", padx=8)
    txt = tk.Text(app, height=14, wrap="word"); txt.pack(fill="both", expand=True, padx=8, pady=4)
    jobs = (b_run, b_dry, b_demo)

    def root_path():
        r = v_root.get().strip()
        if not r:
            messagebox.showerror("wgvp", "choose the top-level data folder first"); return None
        if not Path(r).is_dir():
            messagebox.showerror("wgvp", f"this folder does not exist:\n{r}"); return None
        return Path(r)

    def do_scan():
        r = root_path()
        if r is None:
            return
        write_default_config(r)
        cfg = load_config(r)
        man = scan_root(r, cfg); state["manifest"] = man
        for i in tree.get_children():
            tree.delete(i)
        state["rows"] = {}; state["html"] = None
        for s in man["sessions"]:
            n = tree.insert("", "end", text=s["session_id"], values=(Path(s["dir"]).name, f"{len(s['recordings'])} recording(s), {len(s['videos'])} video(s)" + ("; " + "; ".join(s["flags"]) if s["flags"] else "")))
            for rec in s["recordings"]:
                rn = tree.insert(n, "end", text=rec["recording_id"], values=(Path(rec["csv"]).name, rec.get("flag", "")))
                for v in rec["videos"]:
                    tree.insert(rn, "end", text="video", values=(Path(v["path"]).name, f"task {v['family']}" + (f", tags {v['tags']}" if v.get("tags") else "")))
            for v in s["videos"]:
                vn = tree.insert(n, "end", text=f"panels: {Path(v['path']).name}", values=("detecting ...", ""))
                def _detect(vn=vn, v=v, sid=s["session_id"]):
                    try:
                        from .video_io import detect_panels, panel_preview
                        pan = detect_panels(Path(v["path"]), n_samples=cfg["views"]["n_sample_frames"], black_level=cfg["views"]["black_level"], min_panel_frac=cfg["views"]["min_panel_frac"])
                        prev = Path(r) / "wgvp_output" / sid / "work" / Path(v["path"]).stem / "panels_preview.png"
                        panel_preview(Path(v["path"]), pan["panels"], prev, mode="pixels")
                        txt_ = f"{pan['layout']}: " + ", ".join(f"{p['id']} {p['box']}" for p in pan["panels"])
                        q.put(("tree", (vn, txt_, str(prev))))
                    except Exception as e:
                        q.put(("tree", (vn, f"panel detection failed: {e}", "")))
                threading.Thread(target=_detect, daemon=True).start()
            tree.item(n, open=True); state["rows"][s["session_id"]] = n
        if man["unmatched"]:
            un = tree.insert("", "end", text=f"UNMATCHED ({len(man['unmatched'])})", values=("", "not processed"))
            for u in man["unmatched"][:300]:
                tree.insert(un, "end", text=u["path"], values=(u["reason"], ""))
        log(manifest_text(man))
        if not man["sessions"]:
            log("nothing to process here: the data folder should hold one folder per subject, with that subject's recording CSVs and videos inside (README, 'Expected input').")
        log(f"matching rules: {config_path(r)} (edit and re-scan if something is missing)")
        g = cfg.get("gvhmr", {})
        if g.get("root") and not v_groot.get():
            v_groot.set(g["root"])
        if g.get("python") and not v_gpy.get():
            v_gpy.set(g["python"])

    def save_gvhmr_cfg(r):
        p = config_path(r)
        cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        cfg.setdefault("gvhmr", {}); cfg["gvhmr"]["root"] = v_groot.get().strip(); cfg["gvhmr"]["python"] = v_gpy.get().strip()
        p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    def do_check_gpu():
        from .check_gpu import check, format_rows, run_in_interpreter
        groot = v_groot.get().strip() or None; gpy = v_gpy.get().strip()
        if gpy:
            rc, out = run_in_interpreter(gpy, groot); log(out)
        else:
            log(format_rows(check(groot)))

    def worker(fn, *a, **k):
        def body():
            try:
                res = fn(*a, **k); q.put(("done", res))
            except RunStopped:
                q.put(("done", {"status": "STOPPED"}))
            except NotStarted as e:
                q.put(("box", str(e))); q.put(("done", {"status": "not started", "quiet": True}))
            except Exception as e:
                import traceback
                q.put(("log", traceback.format_exc())); q.put(("done", {"status": "FAIL", "error": str(e)}))
        threading.Thread(target=body, daemon=True).start()

    def begin(job: str, prefix: str = ""):
        """One job at a time: the Stop button is live while it runs, the job buttons are not."""
        state["running"] = True; state["job"] = job; state["stop"].clear()
        for b in jobs:
            b.config(state="disabled")
        b_stop.config(state="normal")
        pb["value"] = 0; state["rp"] = RunProgress(); state["prefix"] = prefix; state["cur"] = None

    def do_stop():
        if not state["running"] or state["stop"].is_set():
            return
        state["stop"].set(); b_stop.config(state="disabled")
        v_prog.set("stopping ... (the step under way ends first; what finished is kept)"); log("stop asked")

    def do_run():
        r = root_path()
        if r is None or state["running"]:
            return
        save_gvhmr_cfg(r)
        from .session import run_root
        begin("run"); v_prog.set("running ...")
        worker(run_root, r, gvhmr_root=v_groot.get().strip() or None, gvhmr_python=v_gpy.get().strip() or None, cpu_only=v_cpu.get(), log=log, progress=progress,
               stop=state["stop"].is_set)

    def do_dry():
        if state["running"]:
            return
        from .dryrun import dry_run
        begin("dry", "dry run, ")
        v_prog.set("dry run on the bundled sample (about a minute, up to six on an older laptop) ...")
        worker(dry_run, out_dir=None, gvhmr_root=v_groot.get().strip() or None, gvhmr_python=v_gpy.get().strip() or None, cpu_only=v_cpu.get(), log=log,
               progress=progress, stop=state["stop"].is_set)

    def demo_job(r: Path, sid: str | None) -> dict:
        """The demo video of every lifted recording of one session (the selected row), or of every lifted session."""
        from .demo import body_models_dir, extras_missing, merged_recordings, merged_sessions
        sessions = merged_sessions(r)
        if not sessions:
            raise NotStarted("no lifted recording under wgvp_output yet: the demo video needs the GPU step (or the lift on the CPU) first")
        if sid and sid not in sessions:
            raise NotStarted(f"{sid} has no lifted recording yet: the demo video needs the GPU step (or the lift on the CPU) first")
        miss = extras_missing()
        if miss:
            raise NotStarted("the demo video needs the demo extras (" + ", ".join(miss) + " missing here):\n    pip install -e .[demo]\nin the pipeline's environment, then try again")
        bm = body_models_dir(load_config(r))
        if not (bm / "smplx").exists():
            raise NotStarted(f"SMPL-X body model files not found under\n{bm}\n(env/install_gvhmr.md, body_models: the GVHMR install's inputs/checkpoints/body_models, named in the GVHMR folder field)")
        from .progress import check_stop
        from .session import stream_child
        made = []
        for s in ([sid] if sid else sessions):
            for rid in merged_recordings(r, s):
                log(f"demo video of {s}/{rid} ...")
                t0 = time.time(); demo_dir = r / "wgvp_output" / s / "work" / "demo"

                def on_line(line):
                    m = re.match(r"\[demo\] (\d+)/(\d+) (\S+)", line)
                    if m:
                        progress("demo", int(m.group(1)), int(m.group(2)), m.group(3))
                    m = re.match(r"demo -> (.+?\.mp4) \(", line)
                    if m:
                        made.append(m.group(1))
                # its own process: the renderer wants the main thread of a process, and a stop ends it outright
                rc = stream_child([sys.executable, "-m", "wgvp", "demo", str(r), "--session", s, "--recording", rid, "--progress"],
                                  lambda line: None if _noise(line) else log(line), stop=state["stop"].is_set, on_line=on_line)
                if state["stop"].is_set():
                    for f in demo_dir.glob("*.mp4"):                     # a hard kill (Windows) can leave the part-written file
                        if f.stat().st_mtime >= t0:
                            f.unlink(missing_ok=True)
                check_stop(state["stop"].is_set)
                if rc != 0:
                    raise RuntimeError(f"the demo video of {s}/{rid} did not render (exit code {rc}); the log above says why")
        return {"demo": made}

    def do_demo():
        r = root_path()
        if r is None or state["running"]:
            return
        sel = tree.focus()
        while sel and tree.parent(sel):                                  # the session the selected row belongs to
            sel = tree.parent(sel)
        sid = tree.item(sel, "text") if sel and sel in state.get("rows", {}).values() else None
        begin("demo"); v_prog.set("demo video: looking for lifted recordings ...")
        worker(demo_job, r, sid)

    def do_open():
        if state.get("html") and Path(state["html"]).exists():
            webbrowser.open(Path(state["html"]).as_uri()); return
        r = root_path()
        if r is None:
            return
        f = r / "wgvp_output" / "QC_SUMMARY.html"
        if f.exists():
            webbrowser.open(f.as_uri())
        else:
            messagebox.showinfo("wgvp", "no QC summary yet - run first")

    def do_open_folder():
        r = root_path()
        if r is None:
            return
        f = r / "wgvp_output"
        f.mkdir(exist_ok=True); webbrowser.open(f.as_uri())

    def show_session_status(only=None):
        """Each finished session's row carries its PASS / WARN / FAIL and the reason (only: one session, as it finishes)."""
        r = Path(v_root.get().strip())
        try:
            from .summary import collect_rows
            why = {}
            for row in collect_rows(r, r / "wgvp_output")[0]:
                for f in row.get("flags") or []:
                    if f not in why.setdefault(row["session"], []):
                        why[row["session"]].append(f)
            for sid, n in state.get("rows", {}).items():
                if only and sid != only:
                    continue
                ss = json.loads((r / "wgvp_output" / sid / "deliver" / "session_summary.json").read_text(encoding="utf-8"))
                flags = why.get(sid) or ([ss["error"]] if ss.get("error") else [])
                tree.set(n, "status", ss.get("status", "?") + (": " + ", ".join(map(str, flags[:2])) if flags else ""))
        except Exception as e:
            log(f"(session results not shown in the list: {e})")

    def finish(payload: dict):
        """The end of a job: the status line, the session rows, the summary or the video to open."""
        t = payload.get("totals"); st = payload.get("stopped")
        if payload.get("quiet"):
            v_prog.set("idle"); return
        if st or payload.get("status") == "STOPPED":
            if isinstance(st, dict):
                v_prog.set(f"stopped after {st['after']} of {st['of']} session(s)" + (f" (PASS {t.get('PASS', 0)}, WARN {t.get('WARN', 0)}, FAIL {t.get('FAIL', 0)} kept)" if t else "")
                           + ". Run all sessions again continues: finished tracking and lifts are reused.")
            else:
                v_prog.set("stopped" + (" (the part-written video was removed)" if state["job"] == "demo" else ""))
            if state.get("cur") in state.get("rows", {}):
                show_session_status(only=state["cur"])
            state["html"] = payload.get("html"); return
        pb["value"] = BAR
        if payload.get("demo"):
            vids = payload["demo"]
            v_prog.set(f"demo video ready: {vids[-1]}" + (f" (and {len(vids) - 1} more, listed in the log)" if len(vids) > 1 else ""))
            webbrowser.open(Path(vids[-1]).as_uri()); return
        if payload.get("status"):                      # the dry run, or a run that raised
            v_prog.set(f"finished: {payload['status']}" + (f" ({payload['error']}; the traceback is in the log below)" if payload.get("error") else ""))
        elif t:
            v_prog.set(f"finished: PASS {t.get('PASS', 0)}, WARN {t.get('WARN', 0)}, FAIL {t.get('FAIL', 0)} of {t.get('sessions', 0)} session(s). Open QC summary shows why.")
        else:
            v_prog.set("finished")
        state["html"] = payload.get("html")
        if t:
            show_session_status()
        elif state.get("cur") in state.get("rows", {}):                      # a run that raised: no result for this session
            tree.set(state["rows"][state["cur"]], "status", "stopped (see the log)")

    def poll():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    txt.insert("end", payload + "\n"); txt.see("end")
                elif kind == "box":
                    messagebox.showinfo("wgvp", payload)
                elif kind == "progress":
                    if payload[0] == "demo":                              # the demo video: one bar per recording, frame by frame
                        frac = payload[1] / max(payload[2], 1); text = f"demo video of {payload[3]}: frame {payload[1]} of {payload[2]}"
                    else:
                        frac, text = state["rp"].update(*payload)            # one bar for the whole run, fed frame by frame
                    if payload[0] == "root" and not state.get("prefix"):   # the list follows the run: the last session's result, this one running
                        if state.get("cur"):
                            show_session_status(only=state["cur"])
                        state["cur"] = payload[3]
                        if payload[3] in state.get("rows", {}):
                            tree.set(state["rows"][payload[3]], "status", "running ...")
                    elif payload[0] in ("lift", "wrap") and state.get("cur") and not state.get("prefix"):
                        show_session_status(only=state["cur"]); state["cur"] = None      # the CPU stages of the last session are done
                    pb["value"] = BAR * frac
                    v_prog.set(("stopping after this step: " if state["stop"].is_set() else "") + f"{state.get('prefix', '')}{text}  ({frac:.0%})")
                elif kind == "tree":
                    vn, txt_, prev = payload
                    try:
                        tree.item(vn, values=(txt_, "double-click to view"))
                        tree.set(vn, "status", prev and "double-click to view")
                        previews[vn] = prev
                    except Exception:
                        pass
                elif kind == "done":
                    state["running"] = False; b_stop.config(state="disabled")
                    for b in jobs:
                        b.config(state="normal")
                    finish(payload if isinstance(payload, dict) else {})
                    state["cur"] = None
        except queue.Empty:
            pass
        app.after(200, poll)

    poll()
    if initial_root:
        app.after(300, do_scan)
    if on_ready:
        on_ready(app)
    app.mainloop()
