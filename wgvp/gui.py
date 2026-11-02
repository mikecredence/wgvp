"""Minimal tkinter GUI: pick the data folder, scan, run, watch progress, open the QC summary. No extra installs."""
from __future__ import annotations

import json
import queue
import threading
import webbrowser
from pathlib import Path

from . import __version__
from .config import config_path, load_config, write_default_config
from .discovery import manifest_text, scan_root


def launch(initial_root: str | None = None):
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox

    app = tk.Tk(); app.title(f"WearGait video pipeline (wgvp {__version__})"); app.geometry("1100x760")
    q: queue.Queue = queue.Queue()
    state = {"manifest": None, "running": False, "summary": None}

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
    tree.column("#0", width=330); tree.column("info", width=520); tree.column("status", width=160)
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
    b_dry = ttk.Button(bar, text="Dry run (bundled sample)", command=lambda: do_dry()); b_dry.pack(side="left", padx=6)
    v_cpu = tk.BooleanVar(value=False); ttk.Checkbutton(bar, text="CPU stages only (skip GVHMR)", variable=v_cpu).pack(side="left", padx=6)
    ttk.Button(bar, text="Open QC summary", command=lambda: do_open()).pack(side="right")
    ttk.Button(bar, text="Open output folder", command=lambda: do_open_folder()).pack(side="right", padx=6)
    pb = ttk.Progressbar(app, mode="determinate"); pb.pack(fill="x", padx=8)
    v_prog = tk.StringVar(value="idle"); ttk.Label(app, textvariable=v_prog).pack(anchor="w", padx=8)
    txt = tk.Text(app, height=14, wrap="word"); txt.pack(fill="both", expand=True, padx=8, pady=4)

    def root_path():
        r = v_root.get().strip()
        if not r or not Path(r).is_dir():
            messagebox.showerror("wgvp", "choose the top-level data folder first"); return None
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
        for s in man["sessions"]:
            n = tree.insert("", "end", text=s["session_id"], values=(Path(s["dir"]).name, f"{len(s['recordings'])} recording(s), {len(s['videos'])} video(s)" + ("; " + "; ".join(s["flags"]) if s["flags"] else "")))
            for rec in s["recordings"]:
                rn = tree.insert(n, "end", text=rec["recording_id"], values=(Path(rec["csv"]).name, rec.get("flag", "")))
                for v in rec["videos"]:
                    tree.insert(rn, "end", text="video", values=(Path(v["path"]).name, f"task {v['family']}" + (f", tags {v['tags']}" if v.get("tags") else "")))
            for v in s["videos"]:
                vn = tree.insert(n, "end", text=f"panels: {Path(v['path']).name}", values=("detecting ...", ""))
                def _detect(vn=vn, v=v):
                    try:
                        from .video_io import detect_panels, panel_preview
                        pan = detect_panels(Path(v["path"]), n_samples=cfg["views"]["n_sample_frames"], black_level=cfg["views"]["black_level"], min_panel_frac=cfg["views"]["min_panel_frac"])
                        prev = Path(r) / "wgvp_output" / s["session_id"] / "work" / Path(v["path"]).stem / "panels_preview.png"
                        panel_preview(Path(v["path"]), pan["panels"], prev, mode="pixels")
                        txt_ = f"{pan['layout']}: " + ", ".join(f"{p['id']} {p['box']}" for p in pan["panels"])
                        q.put(("tree", (vn, txt_, str(prev))))
                    except Exception as e:
                        q.put(("tree", (vn, f"panel detection failed: {e}", "")))
                threading.Thread(target=_detect, daemon=True).start()
            tree.item(n, open=True)
        if man["unmatched"]:
            un = tree.insert("", "end", text=f"UNMATCHED ({len(man['unmatched'])})", values=("", "not processed"))
            for u in man["unmatched"][:300]:
                tree.insert(un, "end", text=u["path"], values=(u["reason"], ""))
        log(manifest_text(man))
        log(f"matching rules: {config_path(r)} (edit and re-scan if something is missing)")
        g = cfg.get("gvhmr", {})
        if g.get("root") and not v_groot.get():
            v_groot.set(g["root"])
        if g.get("python") and not v_gpy.get():
            v_gpy.set(g["python"])

    def save_gvhmr_cfg(r):
        p = config_path(r)
        cfg = json.loads(p.read_text()) if p.exists() else {}
        cfg.setdefault("gvhmr", {}); cfg["gvhmr"]["root"] = v_groot.get().strip(); cfg["gvhmr"]["python"] = v_gpy.get().strip()
        p.write_text(json.dumps(cfg, indent=1))

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
            except Exception as e:
                import traceback
                q.put(("log", traceback.format_exc())); q.put(("done", {"status": "FAIL", "error": str(e)}))
        threading.Thread(target=body, daemon=True).start()

    def do_run():
        r = root_path()
        if r is None or state["running"]:
            return
        save_gvhmr_cfg(r)
        from .session import run_root
        state["running"] = True; b_run.config(state="disabled"); b_dry.config(state="disabled"); v_prog.set("running ...")
        worker(run_root, r, gvhmr_root=v_groot.get().strip() or None, gvhmr_python=v_gpy.get().strip() or None, cpu_only=v_cpu.get(), log=log, progress=progress)

    def do_dry():
        if state["running"]:
            return
        from .dryrun import dry_run
        state["running"] = True; b_run.config(state="disabled"); b_dry.config(state="disabled"); v_prog.set("dry run ...")
        worker(dry_run, out_dir=None, gvhmr_root=v_groot.get().strip() or None, gvhmr_python=v_gpy.get().strip() or None, cpu_only=v_cpu.get(), log=log)

    def do_open():
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

    def poll():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    txt.insert("end", payload + "\n"); txt.see("end")
                elif kind == "progress":
                    stage, i, n, msg = payload
                    if stage == "root":
                        pb["maximum"] = max(n, 1); pb["value"] = i
                    v_prog.set(f"{stage}: {i + 1}/{n} {msg}")
                elif kind == "tree":
                    vn, txt_, prev = payload
                    try:
                        tree.item(vn, values=(txt_, "double-click to view"))
                        tree.set(vn, "status", prev and "double-click to view")
                        previews[vn] = prev
                    except Exception:
                        pass
                elif kind == "done":
                    state["running"] = False; b_run.config(state="normal"); b_dry.config(state="normal")
                    st = (payload or {}).get("status") or (payload or {}).get("totals")
                    v_prog.set(f"finished: {st}"); pb["value"] = pb["maximum"]
                    if isinstance(payload, dict) and payload.get("html"):
                        log(f"QC summary: {payload['html']}")
        except queue.Empty:
            pass
        app.after(200, poll)

    poll()
    if initial_root:
        app.after(300, do_scan)
    app.mainloop()
