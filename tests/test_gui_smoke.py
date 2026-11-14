"""The window itself, driven the way a user drives it: scan a folder, check the GPU environment, start a run and stop it,
ask for a demo video where nothing is lifted, run to the end, read the result, open the summary, run the one session
again by itself.

It needs a display and about a minute, so it runs only with WGVP_GUI_SMOKE=1 (the CI sets it on Windows, macOS
and Linux). The data folder has a space and an accent in its name, as a folder on a real PC may."""
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _widgets(w):
    for c in w.winfo_children():
        yield c
        yield from _widgets(c)


def _find(app, cls, text=None):
    for w in _widgets(app):
        if w.winfo_class() == cls and (text is None or str(w.cget("text")) == text):
            return w
    raise AssertionError(f"no {cls} {text!r} in the window")


@pytest.mark.skipif(os.environ.get("WGVP_GUI_SMOKE") != "1", reason="set WGVP_GUI_SMOKE=1 to drive the real window")
def test_scan_run_and_result_in_the_real_window(tmp_path, monkeypatch):
    pytest.importorskip("tkinter")
    import webbrowser
    from tkinter import messagebox

    from wgvp import gui

    root = tmp_path / "data root \u00e9t\u00e9"
    shutil.copytree(REPO / "examples" / "bundled" / "OWN01", root / "OWN01")
    shutil.copy(REPO / "examples" / "bundled" / "wgvp_config.json", root)
    seen = {"boxes": [], "opened": [], "bar": [], "status": "", "row": None, "error": None}
    monkeypatch.setattr(messagebox, "showerror", lambda title, msg, **k: seen["boxes"].append(msg))
    monkeypatch.setattr(messagebox, "showinfo", lambda title, msg, **k: seen["boxes"].append(msg))
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: seen["opened"].append(url))

    def script(app):
        tree, bar = _find(app, "Treeview"), _find(app, "TProgressbar")
        label = next(w for w in _widgets(app) if w.winfo_class() == "TLabel" and str(w.cget("textvariable")))
        status = lambda: str(app.getvar(str(label.cget("textvariable"))))
        rows = lambda: [(i, tree.item(i, "text"), [str(v) for v in tree.item(i, "values")]) for s in tree.get_children() for i in (s, *tree.get_children(s))]
        t0 = time.time()                                           # the window scans its starting folder by itself
        while not rows() or any("detecting" in " ".join(v) for _, _, v in rows()):
            assert time.time() - t0 < 180, f"the scan did not finish: {rows()}"
            yield 200
        assert [t for _, t, _ in rows() if t == "OWN01"], rows()
        entries = [w for w in _widgets(app) if w.winfo_class() == "TEntry"]      # data folder, GVHMR folder, GVHMR python
        app.setvar(str(entries[2].cget("textvariable")), sys.executable)        # no GVHMR in this interpreter: the check must say so
        ticks = {"n": 0, "on": True}                                             # and the window must keep painting meanwhile

        def tick():
            if ticks["on"]:
                ticks["n"] += 1; app.after(20, tick)
        app.after(20, tick); t0 = time.time()
        _find(app, "TButton", "Check GPU").invoke()
        seen["gpu_busy"] = (status(), tuple(str(_find(app, "TButton", b).cget("state")) for b in ("Run all sessions", "Stop", "Check GPU")))
        while not status().startswith("GPU stage"):
            assert time.time() - t0 < 180, f"the GPU check did not end: {status()}"
            yield 100
        ticks["on"] = False; seen["gpu"] = (status(), time.time() - t0, ticks["n"]); seen["gpu_log"] = _find(app, "Text").get("1.0", "end")
        app.setvar(str(entries[2].cget("textvariable")), "")
        _find(app, "TCheckbutton").invoke()                        # CPU stages only
        _find(app, "TButton", "Run all sessions").invoke()         # a run, stopped while it tracks
        t0 = time.time()
        while float(bar.cget("value")) < 0.12 * gui.BAR:
            assert time.time() - t0 < 600, f"the run did not get going: {status()}"
            yield 200
        _find(app, "TButton", "Stop").invoke()
        t0 = time.time()
        while not status().startswith("stopped"):
            assert time.time() - t0 < 120, f"the run did not stop: {status()}"
            yield 200
        yield 400
        seen["stop_status"] = status(); seen["stop_row"] = next(v for _, t, v in rows() if t == "OWN01")
        seen["buttons"] = tuple(str(_find(app, "TButton", b).cget("state")) for b in ("Run all sessions", "Stop", "Demo video"))
        _find(app, "TButton", "Demo video").invoke()                # nothing is lifted here: a dialog says so and nothing runs
        t0 = time.time()
        while not seen["boxes"] or status() != "idle":
            assert time.time() - t0 < 60, f"no answer to the demo button: {status()} {seen['boxes']}"
            yield 200
        seen["demo_box"] = list(seen["boxes"])
        _find(app, "TButton", "Run all sessions").invoke()         # the run, to the end
        t0 = time.time()
        while not status().startswith("finished"):
            assert time.time() - t0 < 900, f"the run did not finish: {status()}"
            seen["bar"].append(float(bar.cget("value")))
            yield 200
        yield 400
        seen["bar"].append(float(bar.cget("value"))); seen["status"] = status()
        seen["row"] = next(v for _, t, v in rows() if t == "OWN01")
        _find(app, "TButton", "Open QC summary").invoke()
        _find(app, "TButton", "Run selected session").invoke()     # nothing selected: a dialog, nothing runs
        yield 300
        seen["one_box"] = list(seen["boxes"]); own = next(i for i, t, _ in rows() if t == "OWN01")
        tree.selection_set(own); tree.focus(own)
        _find(app, "TButton", "Run selected session").invoke()     # the one session by itself (its tracking is cached, so quick)
        seen["one_start"] = status(); t0 = time.time()
        while not status().startswith("finished"):
            assert time.time() - t0 < 600, f"the single-session run did not finish: {status()}"
            yield 200
        yield 400
        seen["one_status"] = status()

    def on_ready(app):
        gen = script(app)

        def pump():
            try:
                app.after(int(next(gen)), pump)
            except StopIteration:
                app.destroy()
            except BaseException:
                seen["error"] = traceback.format_exc(); app.destroy()
        app.after(1000, pump)

    gui.launch(str(root), on_ready=on_ready)
    assert seen["error"] is None, seen["error"]
    assert seen["gpu_busy"][0].startswith("checking the GPU environment of ") and seen["gpu_busy"][1] == ("disabled", "normal", "disabled"), seen["gpu_busy"]
    st, el, ticks = seen["gpu"]
    assert st.startswith("GPU stage NOT ready: ") and st.endswith("the log below says what is missing and how to fix it"), st
    assert el < 0.5 or ticks >= 5 * el, (el, ticks)                # the window kept painting while the check ran in another interpreter
    assert "[MISSING]" in seen["gpu_log"] and "GPU stage NOT ready" in seen["gpu_log"], seen["gpu_log"][-800:]
    assert seen["stop_status"].startswith("stopped after 0 of 1 session(s)"), seen["stop_status"]
    assert seen["stop_row"][-1].startswith("STOPPED: stopped by the user"), seen["stop_row"]
    assert seen["buttons"] == ("normal", "disabled", "normal"), seen["buttons"]
    assert len(seen["demo_box"]) == 1 and seen["demo_box"][0].startswith("no lifted recording under wgvp_output yet"), seen["demo_box"]
    assert seen["one_box"] == seen["demo_box"] + ["click a session in the list first (any of its rows), then Run selected session"], seen["one_box"]
    assert seen["boxes"] == seen["one_box"], seen["boxes"]         # no other dialog came up
    assert seen["one_start"] == "running OWN01 ...", seen["one_start"]
    assert seen["one_status"].startswith(("finished: OWN01 PASS.", "finished: OWN01 WARN.")) and "of 1)" in seen["one_status"], seen["one_status"]
    # PASS on macOS and Windows; on Linux the same clip decodes a little differently and ends WARN (flash_disagrees)
    assert seen["status"].startswith(("finished: PASS 1, WARN 0, FAIL 0 of 1", "finished: PASS 0, WARN 1, FAIL 0 of 1")), seen["status"]
    assert seen["row"][-1].startswith(("PASS", "WARN: ")), seen["row"]
    bar = seen["bar"]
    assert bar == sorted(bar) and bar[-1] == gui.BAR, bar[-5:]     # the bar never goes back and ends full
    assert len(set(bar)) >= 8, sorted(set(bar))                    # and it moves through the run, not in one jump
    assert seen["opened"] and seen["opened"][-1].endswith("QC_SUMMARY.html"), seen["opened"]
    assert (root / "wgvp_output" / "QC_SUMMARY.html").exists()
