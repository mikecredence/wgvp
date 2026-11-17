"""The README's window clip, on macOS: record the window through a scan and a CPU-stages run, then build the clip.

    python tools/record_window_mac.py record <frames folder> <data folder>     # frames + frames.json
    python tools/record_window_mac.py clip <frames folder> <out folder>        # gui_preview.webp (1220 px) and gui.png

The window photographs itself through Quartz (its own buffer, found by this process's id and the window's title),
so no screen-recording permission is needed and the window may be covered meanwhile. The data folder should sit
under a neutral path (the field and the log show it); the three bundled sessions give PASS 1, WARN 2 in two minutes.
Needs the demo extras' pyglet (pip install -e .[demo]) and Pillow.
"""
import json
import os
import sys
import time
import traceback
import webbrowser
from ctypes import c_uint32, c_void_p, string_at
from pathlib import Path
from tkinter import messagebox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W = 1220

from pyglet.libs.darwin.cocoapy import CFSTR, CGRect, cf, cfnumber_to_number, cfstring_to_string, quartz

quartz.CGWindowListCopyWindowInfo.restype = c_void_p
quartz.CGWindowListCopyWindowInfo.argtypes = [c_uint32, c_uint32]
quartz.CGWindowListCreateImage.restype = c_void_p
quartz.CGWindowListCreateImage.argtypes = [CGRect, c_uint32, c_uint32, c_uint32]
cf.CFDictionaryGetValue.restype = c_void_p
cf.CFDictionaryGetValue.argtypes = [c_void_p, c_void_p]
ALL_WINDOWS, INCLUDING_WINDOW, IGNORE_FRAMING, BEST_RESOLUTION = 0, 1 << 3, 1 << 0, 1 << 3
K_PID, K_NUMBER, K_NAME, K_LAYER = CFSTR("kCGWindowOwnerPID"), CFSTR("kCGWindowNumber"), CFSTR("kCGWindowName"), CFSTR("kCGWindowLayer")


def window_number(title: str) -> int:
    """The CGWindowID of this process's window whose title contains `title`."""
    arr = quartz.CGWindowListCopyWindowInfo(ALL_WINDOWS, 0); pid = os.getpid(); found = []
    for i in range(cf.CFArrayGetCount(arr)):
        d = cf.CFArrayGetValueAtIndex(arr, i)
        if cfnumber_to_number(cf.CFDictionaryGetValue(d, K_PID)) != pid:
            continue
        name = cf.CFDictionaryGetValue(d, K_NAME); name = cfstring_to_string(name) if name else ""
        found.append((name, int(cfnumber_to_number(cf.CFDictionaryGetValue(d, K_NUMBER)))))
    cf.CFRelease(arr)
    for name, num in found:
        if title in name:
            return num
    raise RuntimeError(f"no window of this process titled like {title!r}; found {found}")


def capture(app, path, shadow=False):
    from PIL import Image
    app.update_idletasks(); app.update()
    r = CGRect(); r.origin.x = float("inf"); r.origin.y = float("inf"); r.size.width = 0.0; r.size.height = 0.0   # CGRectNull: the window's own bounds
    img = quartz.CGWindowListCreateImage(r, INCLUDING_WINDOW, window_number(app.title()), (0 if shadow else IGNORE_FRAMING) | BEST_RESOLUTION)
    if not img:
        raise RuntimeError("CGWindowListCreateImage returned nothing")
    w, h, bpr = quartz.CGImageGetWidth(img), quartz.CGImageGetHeight(img), quartz.CGImageGetBytesPerRow(img)
    data = quartz.CGDataProviderCopyData(quartz.CGImageGetDataProvider(img))
    buf = string_at(cf.CFDataGetBytePtr(data), cf.CFDataGetLength(data))
    im = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", bpr, 1).convert("RGB")
    cf.CFRelease(data); quartz.CGImageRelease(img)
    im.save(path)
    return w, h


frames, err = [], []
OUT = Path()


def _widgets(w):
    for c in w.winfo_children():
        yield c
        yield from _widgets(c)


def _find(app, cls, text=None):
    for w in _widgets(app):
        if w.winfo_class() == cls and (text is None or str(w.cget("text")) == text):
            return w
    raise AssertionError(f"no {cls} {text!r}")


def shot(app, tag):
    p = OUT / f"f{len(frames):03d}_{tag}.png"; capture(app, p); frames.append({"t": time.time(), "tag": tag, "file": p.name}); print("shot", p.name, flush=True)


def script(app):
    app.tk.call("wm", "attributes", ".", "-appearance", "light"); app.geometry("1220x760+60+80"); app.lift(); app.update()
    tree = _find(app, "Treeview")
    label = next(w for w in _widgets(app) if w.winfo_class() == "TLabel" and str(w.cget("textvariable")))
    status = lambda: str(app.getvar(str(label.cget("textvariable"))))
    rows = lambda: [(tree.item(i, "text"), [str(v) for v in tree.item(i, "values")]) for s in tree.get_children() for i in (s, *tree.get_children(s))]
    yield 300; shot(app, "start")
    t0 = time.time(); last = time.time()
    while not rows() or any("detecting" in " ".join(v) for _, v in rows()):
        assert time.time() - t0 < 180, "scan"
        if time.time() - last > 1.5:
            shot(app, "scan"); last = time.time()
        yield 200
    yield 500; shot(app, "scanned"); yield 700
    _find(app, "TCheckbutton").invoke()                               # CPU stages only
    _find(app, "TButton", "Run all sessions").invoke()
    t0 = time.time(); yield 300; shot(app, "run")
    while not status().startswith("finished"):
        assert time.time() - t0 < 900, "run"
        yield 2000; shot(app, "run")
    yield 600; shot(app, "finished"); yield 1200; shot(app, "finished")
    (OUT / "frames.json").write_text(json.dumps({"frames": frames, "status": status(), "rows": rows()}, indent=1), encoding="utf-8")


def on_ready(app):
    gen = script(app)

    def pump():
        try:
            app.after(int(next(gen)), pump)
        except StopIteration:
            app.destroy()
        except BaseException:
            err.append(traceback.format_exc()); app.destroy()
    app.after(800, pump)



def record(out: Path, data: str) -> int:
    global OUT
    OUT = out; out.mkdir(parents=True, exist_ok=True)
    webbrowser.open = lambda *a, **k: None
    messagebox.showinfo = messagebox.showerror = lambda *a, **k: None
    from wgvp import gui
    gui.launch(data, on_ready=on_ready)
    print("\n".join(err) if err else f"recorded {len(frames)} frames", flush=True)
    return 1 if err else 0


def build_clip(rec: Path, out: Path) -> int:
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True); REC, OUT = rec, out
    meta = json.loads((REC / "frames.json").read_text(encoding="utf-8")); fr = meta["frames"]
    by = lambda tag: [f for f in fr if f["tag"] == tag]
    run = by("run"); step = max(1, round(len(run) / 24))
    chosen = ([(f, 1300) for f in by("start")] + [(f, 650) for f in by("scan")[:4]] + [(f, 1600) for f in by("scanned")]
              + [(f, 600) for f in run[::step]] + [(f, 2500) for f in by("finished")[:1]] + [(f, 3200) for f in by("finished")[1:2]])
    ims = []
    for f, d in chosen:
        im = Image.open(REC / f["file"]).convert("RGB"); h = round(im.height * W / im.width); ims.append((im.resize((W, h), Image.LANCZOS), d))
    first = ims[0][0]
    first.save(OUT / "gui_preview.webp", save_all=True, append_images=[i for i, _ in ims[1:]], duration=[d for _, d in ims], loop=0, quality=80, method=6)
    ims[-1][0].save(OUT / "gui.png", optimize=True)
    tot = sum(d for _, d in ims) / 1000
    print(f"{len(ims)} frames, {tot:.1f} s, webp {(OUT / 'gui_preview.webp').stat().st_size // 1024} KB, png {(OUT / 'gui.png').stat().st_size // 1024} KB, run span {run[-1]['t'] - run[0]['t']:.0f} s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("record", "clip"):
        print(__doc__); sys.exit(2)
    sys.exit(record(Path(sys.argv[2]), sys.argv[3]) if sys.argv[1] == "record" else build_clip(Path(sys.argv[2]), Path(sys.argv[3])))
