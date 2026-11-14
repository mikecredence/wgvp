"""Demo video of a lifted recording, rendered from the pipeline's own outputs: the two views with the lifter's 2D
input and what each view can use, the reconstruction (red = a part no view could see), the export (what a user
takes away: parts no view saw are absent, dropped frames show the empty floor), an aerial view, and a readout
(a filmstrip of small renders with the well-tracked blocks marked, and knee-flexion timelines).

    python -m wgvp demo <root> [--session ID] [--recording ID] [--gt <sample>_gt.npz] [--t0 S --t1 S] [--step N]
                               [--gif FILE --gif-span A,B] [--readout both|charts|strips] [--body-models DIR] [--out FILE]

Inputs: <session>/deliver/<recording>/{merged_smpl.npz, sync.json, <view>/smpl_global.npz, <view>/kp2d_vitpose.npz}
and the session video. A ground truth (`--gt`, the rendered samples' <sid>_gt.npz) adds the truth panel, row and
curve and the leg-joint and root-orientation errors; when the truth carries its room, the three close-ups stand
in that room (the truth's position, floor and mat; the reconstruction turned into it by one yaw fitted over the
recording), so a heading difference on screen is a heading error. Without a truth the aerial view shows the
reconstruction on its own path.
Output: <session>/work/demo/<recording>_demo.mp4. It contains video frames, so it stays in work/ and is not a
deliverable. Needs the demo extras (`pip install -e .[demo]`: smplx, trimesh, pyrender, av, matplotlib) and the
SMPL-X body model (`--body-models`, default: GVHMR's inputs/checkpoints/body_models from the config).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

from .fusion import ARMS, LEGS, SPINE, chordal_mean, collapsed_legs, fit_yaw, geodesic, mirror_body, mirror_root, seen_frames, smooth_rotations
from .merge import path_slope_deg
from .progress import RunStopped, check_stop
from .tracking import FACE_KP, SKELETON

COL = {"ok": [150, 160, 200], "missing": [220, 70, 70], "truth": [170, 170, 170]}
VIEW_BGR = [(180, 119, 31), (14, 127, 255), (160, 60, 160)]      # per view, in `views` order (tab:blue, tab:orange, purple)
VIEW_MPL = ["tab:blue", "tab:orange", "tab:purple"]
MERGE_BGR = (63, 209, 63)
SMOOTH_FRAMES = 5


def knee_deg(body):   # (T, 21, 3) -> (T, 2) left, right knee flexion magnitude
    return np.degrees(np.linalg.norm(body[:, [3, 4]], axis=2))


def side_consistent(kp: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Display-only: COCO-17 keypoints (T, 17, 3) with the left/right labels of the legs and arms swapped in a frame
    when the swapped assignment continues the previous frame more closely. 2D pose estimators flip sides on a
    side-on walker; the 3D stage does not use these labels, so the flips only matter for the overlay."""
    kp = kp.copy(); legs = [11, 13, 15]; arms = [5, 7, 9]
    for chain in (legs, arms):
        L, Rr = chain, [i + 1 for i in chain]; prev = None; since = 0
        for t in range(len(kp)):
            a, b = kp[t, L, :2], kp[t, Rr, :2]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                since += 1; continue
            if prev is not None and since < max_gap:
                pa, pb = prev
                keep = np.linalg.norm(a - pa, axis=1).sum() + np.linalg.norm(b - pb, axis=1).sum()
                swap = np.linalg.norm(b - pa, axis=1).sum() + np.linalg.norm(a - pb, axis=1).sum()
                if swap < 0.7 * keep:
                    kp[t, L], kp[t, Rr] = kp[t, Rr].copy(), kp[t, L].copy(); a, b = kp[t, L, :2], kp[t, Rr, :2]
            prev = (a.copy(), b.copy()); since = 0
    return kp


def merged_sessions(root: Path) -> list[str]:
    """The sessions under <root>/wgvp_output that hold a merged recording (deliver/<recording>/merged_smpl.npz)."""
    return sorted({d.parents[2].name for d in (Path(root) / "wgvp_output").glob("*/deliver/*/merged_smpl.npz")})


def merged_recordings(root: Path, session: str) -> list[str]:
    """The merged recordings of one session, by recording id."""
    return sorted(p.parent.name for p in (Path(root) / "wgvp_output" / session / "deliver").glob("*/merged_smpl.npz"))


def body_models_dir(cfg: dict) -> Path:
    """Where the SMPL-X files are looked for: the GVHMR install's body_models, else ~/data/WearGait/body_models."""
    cand = Path(cfg.get("gvhmr", {}).get("root") or "") / "inputs" / "checkpoints" / "body_models"
    return cand if (cand / "smplx").exists() else Path.home() / "data" / "WearGait" / "body_models"


def extras_missing() -> list[str]:
    """The demo's optional packages that do not import here (pip install -e .[demo])."""
    import importlib
    out = []
    for m in ("torch", "smplx", "trimesh", "pyrender", "av", "matplotlib"):
        try:
            importlib.import_module(m)
        except Exception:
            out.append(m)
    return out


def _mac_screen_fallback() -> None:
    """macOS lists no ACTIVE display while the screen is asleep or locked; pyglet then finds no screen and pyrender
    cannot open its context. The ONLINE list still names the display, and an offscreen context works on it."""
    import sys
    if sys.platform != "darwin":
        return
    from ctypes import byref, c_uint32
    import pyglet.display.cocoa as pc
    if getattr(pc.CocoaDisplay, "_wgvp_fallback", False):
        return
    active = pc.CocoaDisplay.get_screens

    def get_screens(self):
        screens = active(self)
        if screens:
            return screens
        ids = (pc.CGDirectDisplayID * 256)(); n = c_uint32(); pc.quartz.CGGetOnlineDisplayList(256, ids, byref(n))
        return [pc.CocoaScreen(self, d) for d in list(ids)[:n.value]]
    pc.CocoaDisplay.get_screens = get_screens; pc.CocoaDisplay._wgvp_fallback = True


def _offscreen(pyrender, w: int, h: int):
    """An offscreen renderer, or a plain sentence when this machine cannot open an OpenGL context."""
    try:
        return pyrender.OffscreenRenderer(w, h)
    except Exception as e:
        raise RuntimeError(f"the demo's renderer could not open an OpenGL context on this machine ({type(e).__name__}: {e}); "
                           "a desktop session with a display is needed (on a headless Linux box, PYOPENGL_PLATFORM=egl with an NVIDIA driver)") from e


def _video_path(root: Path, deliver: Path, rid: str) -> Path | None:
    summ = json.loads((deliver / "session_summary.json").read_text(encoding="utf-8")) if (deliver / "session_summary.json").exists() else {}
    vid = (summ.get("items", {}).get(rid) or {}).get("video_id"); vrec = (summ.get("videos") or {}).get(vid or "", {})
    p = vrec.get("file") or vrec.get("path")
    if p and Path(p).exists():
        return Path(p)
    man = root / "wgvp_output" / "manifest.json"
    if man.exists():
        for s in json.loads(man.read_text(encoding="utf-8")).get("sessions", []):
            for r in s.get("recordings", []):
                if r.get("recording_id") == rid:
                    for v in r.get("videos", []):
                        if Path(v["path"]).exists():
                            return Path(v["path"])
    return None


def resolve_target(rdir: Path, target: str = "auto") -> tuple[str, str, str]:
    """The pose a demo shows -> (method, file name, suffix of the demo's file name). "auto" follows the recording's
    target.json (the merge when there is none); "merge" / "jointfit" ask for one by name and mark the file name."""
    if target == "auto":
        try:
            method = json.loads((Path(rdir) / "target.json").read_text(encoding="utf-8")).get("method") or "merge"
        except (OSError, ValueError):
            method = "merge"
        return method, "jointfit_smpl.npz" if method == "jointfit" else "merged_smpl.npz", ""
    return target, "jointfit_smpl.npz" if target == "jointfit" else "merged_smpl.npz", f"_{target}"


def render_demo(root: Path, session: str, recording: str | None = None, gt: str | None = None, out: str | None = None, body_models: str | None = None,
                t0: float | None = None, t1: float | None = None, step: int = 1, gif: str | None = None, gif_span: str | None = None,
                readout: str = "both", strip_step: float = 2.0, log=print, target: str = "auto", stop=None, progress=None) -> dict:
    """Render the demo video of one lifted recording. stop (a callable) ends the render at the next frame and removes the
    part-written file; progress("demo", frame, n_frames, recording) is called every few frames."""
    import cv2
    from scipy.spatial.transform import Rotation as R
    from .config import load_config
    from .session import session_out
    root = Path(root); cfg = load_config(root); sdir = session_out(root, session); deliver = sdir / "deliver"
    recs = sorted(p.parent.name for p in deliver.glob("*/merged_smpl.npz"))
    if not recs:
        raise SystemExit(f"no merged recording under {deliver} (run the GPU stage, then `wgvp merge`)")
    rid = recording or recs[0]; rdir = deliver / rid
    video = _video_path(root, deliver, rid)
    if video is None:
        raise SystemExit(f"video of {session}/{rid} not found (session_summary.json / manifest.json)")
    if body_models is None:
        body_models = str(body_models_dir(cfg))
    target, tfile, suffix = resolve_target(rdir, target); tname = "joint fit" if target == "jointfit" else "merge"     # the pose shown: the recording's training target unless one is asked for by name
    out = Path(out) if out else sdir / "work" / "demo" / f"{rid}_demo{suffix}.mp4"; out.parent.mkdir(parents=True, exist_ok=True)
    M = np.load(rdir / "merged_smpl.npz"); views = [str(v) for v in M["views"]]; V = len(views)
    if target == "jointfit" and not (rdir / tfile).exists():
        raise FileNotFoundError(f"{rdir / tfile}: run `wgvp jointfit` first")
    S = np.load(rdir / tfile) if target == "jointfit" else M
    Z = {v: np.load(rdir / v / "smpl_global.npz") for v in views}; K = {v: np.load(rdir / v / "kp2d_vitpose.npz") for v in views}
    offset = float(json.loads((rdir / "sync.json").read_text(encoding="utf-8"))["offset_s"])
    cap = cv2.VideoCapture(str(video)); n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(M["fps"]) if "fps" in M.files else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    t_all = np.arange(n_video) / fps - offset
    G = np.load(gt) if gt else None; gi = {int(f): i for i, f in enumerate(G["frame_idx"])} if G is not None else {}
    exp = {}
    if gt and (Path(gt).parent / "expected.json").exists():
        exp = json.loads((Path(gt).parent / "expected.json").read_text(encoding="utf-8"))
    # ---- per view, dense over the video's frames
    body, root_al, usable, lifted, vis, kp_draw, held = {}, {}, {}, {}, {}, {}, {}
    yaw_al = {v: float(y) for v, y in zip(views, M["view_alignment_yaw_deg"])}
    slope = {v: s_ for v in views if (s_ := path_slope_deg(Z[v]["transl"], fps)) is not None} if G is None else {}      # no truth: the level floor is a check that needs none
    mirrored = {v: bool(m) for v, m in zip(views, M["view_mirrored"])} if "view_mirrored" in M.files else {v: False for v in views}
    collapsed = {}
    for i, v in enumerate(views):
        z, k = Z[v], K[v]; fr = z["frame_idx"].astype(int)
        body[v] = np.full((n_video, 21, 3), np.nan); root_al[v] = np.full((n_video, 3), np.nan); lifted[v] = np.zeros(n_video, bool); vis[v] = np.zeros(n_video, bool)
        bp = z["body_pose"].reshape(-1, 21, 3); ro = z["global_orient"]
        if mirrored[v]:                                                    # the merge reflected this view back; show it the same way
            bp = mirror_body(bp); ro = mirror_root(ro)
        body[v][fr] = bp; lifted[v][fr] = True; vis[v][fr] = seen_frames(z, k)       # seen: detected, or held at rest and confirmed
        held[v] = np.zeros(n_video, bool); held[v][fr] = vis[v][fr] & ~z["visible"].astype(bool)
        with np.errstate(all="ignore"):
            root_al[v][fr] = (R.from_euler("y", yaw_al[v], degrees=True) * R.from_rotvec(ro)).as_rotvec()
        arr = np.full((n_video, 17, 3), np.nan); arr[k["frame_idx"].astype(int)] = k["kp2d"]; kp_draw[v] = side_consistent(arr)
        collapsed[v] = np.zeros(n_video, bool); collapsed[v][k["frame_idx"].astype(int)] = collapsed_legs(k["kp2d"])
    mf = M["frame_idx"].astype(int)
    usable_all = np.zeros((V, n_video, 21), bool); usable_all[:, mf] = M["usable"]
    for i, v in enumerate(views):
        usable[v] = usable_all[i]
    share = np.zeros((V, n_video)); share[:, mf] = M["share"]
    merged_out = np.full((n_video, 21, 3), np.nan); merged_out[mf] = S["body_pose"].reshape(-1, 21, 3)
    root_fused = np.full((n_video, 3), np.nan); root_fused[mf] = S["global_orient"]
    transl_f = np.full((n_video, 3), np.nan); transl_f[mf] = S["transl"]
    export_ok = np.zeros(n_video, bool); export_ok[mf] = M["export_ok"]
    any_lifted = np.any([lifted[v] for v in views], axis=0); none = ~np.any([usable[v] for v in views], axis=0)
    # the drawn reconstruction: the stored pose as it is stored; unseen parts (shown red) filled from whatever was lifted, with the merge's smoothing
    est = [np.nan_to_num(body[v]) for v in views]
    fill = chordal_mean(est, np.stack([lifted[v][:, None].repeat(21, 1).astype(float) for v in views]))
    md = np.where(none[:, :, None], fill, np.nan_to_num(merged_out)); md[~any_lifted] = np.nan
    merged_draw = np.where(none[:, :, None] | ~np.isfinite(merged_out), np.nan_to_num(smooth_rotations(md, SMOOTH_FRAMES)), np.nan_to_num(merged_out))
    leg_ok = {v: usable[v][:, LEGS].all(1) for v in views}
    # ---- truth series
    truth_body = np.full((n_video, 21, 3), np.nan)
    for f, i in gi.items():
        if f < n_video:
            truth_body[f] = G["pose_aa"][i, 3:66].reshape(21, 3)
    kn = {v: knee_deg(body[v]) for v in views}; kn["merge"] = knee_deg(merged_out); kn["truth"] = knee_deg(truth_body)
    for v in views:                                                             # a view's readout is only what it would output: its usable joints
        kn[v][~usable[v][:, 3], 0] = np.nan; kn[v][~usable[v][:, 4], 1] = np.nan
    # ---- frames to draw
    span = t_all[any_lifted] if any_lifted.any() else t_all
    t0 = float(span.min()) if t0 is None else t0; t1 = float(span.max()) if t1 is None else t1
    frames = [f for f in range(n_video) if t0 <= t_all[f] <= t1][::max(step, 1)]
    # ---- SMPL-X + renderer
    if "PYOPENGL_PLATFORM" not in os.environ or os.environ["PYOPENGL_PLATFORM"] != "egl":
        os.environ.pop("PYOPENGL_PLATFORM", None)
    _mac_screen_fallback()
    import torch, smplx, trimesh, pyrender
    bm = smplx.create(body_models, model_type="smplx", gender="neutral", use_pca=False, batch_size=1); faces = bm.faces.astype(np.int32)
    dom = bm.lbs_weights.detach().cpu().numpy()[:, :22].argmax(1)
    def verts(root_aa, body_aa, transl=None):
        with torch.no_grad():
            o = bm(global_orient=torch.tensor(root_aa, dtype=torch.float32)[None], body_pose=torch.tensor(body_aa.reshape(63), dtype=torch.float32)[None], betas=torch.zeros(1, 10),
                   transl=torch.zeros(1, 3) if transl is None else torch.tensor(transl, dtype=torch.float32)[None])
        return o.vertices[0].numpy()
    RW, RH = 340, 442; AW, AH = 360, 250
    renderer = _offscreen(pyrender, RW, RH); aerial_r = _offscreen(pyrender, AW, AH)
    # the aerial scene: with a truth, the sample's room (mat, cameras, operator, bystander) from expected.json; without one, the
    # reconstruction's own path in the reference view's world
    if G is not None:
        lay0 = exp.get("scene", {}).get("layout") or {}
        if "mat_x" in lay0:                                                    # the room's mat as rendered (16 x 2 ft, centred on the walk)
            x0m, x1m = [float(v) for v in lay0["mat_x"]]; zm = float(np.mean(lay0["mat_z"])); mat_half_w = 0.5 * float(lay0["mat_z"][1] - lay0["mat_z"][0]); mat_pad = 0.0
        else:
            x0m, x1m = exp.get("scene", {}).get("walk_extent_m", [-2.45, 2.45]); zm = float(np.median(G["transl"][:, 2])) if "transl" in G.files else 0.0; mat_half_w = 0.5; mat_pad = 0.8
        cams = {v: G[f"{v}_cam_pose"] for v in views if f"{v}_cam_pose" in G.files}; ext = np.zeros(3)
    else:
        okt = np.isfinite(transl_f).all(1); ctr = np.median(transl_f[okt], axis=0) if okt.any() else np.zeros(3)
        ext = np.nanmax(np.abs(transl_f[okt] - ctr), axis=0) if okt.any() else np.array([2.0, 0.0, 2.0])
        x0m, x1m = float(ctr[0] - ext[0]), float(ctr[0] + ext[0]); zm = float(ctr[2]); cams = {}; mat_half_w = 0.5; mat_pad = 0.8
    xm_ = 0.5 * (x0m + x1m)
    lay = exp.get("scene", {}).get("layout") if G is not None else None
    # With a truth that carries its room (the root path and the mat), the close-ups are drawn in that room: the truth where it
    # walked, the reconstruction and the export turned into the room by ONE yaw fitted over the whole recording and stood at the
    # truth's position. The merge knows neither the room's heading (its world is the reference view's, with an arbitrary heading)
    # nor, to better than a few decimetres, where in the room the subject is; what is left to see is the pose and the heading error.
    room = bool(G is not None and "transl" in G.files and "mat_x" in lay0)
    truth_root = np.full((n_video, 3), np.nan); truth_xz = np.full((n_video, 2), np.nan); yaw_room = 0.0
    for f_, i_ in gi.items():
        if f_ < n_video:
            truth_root[f_] = G["pose_aa"][i_, :3]
            if room:
                truth_xz[f_] = G["transl"][i_][[0, 2]]
    if room:
        okf = export_ok & np.isfinite(root_fused).all(1) & np.isfinite(truth_root).all(1)
        if okf.sum() >= 10:
            yaw_room = float(fit_yaw(root_fused, truth_root, okf))
    def to_room(aa):
        """Root rotations in the reference view's world -> the room's (one yaw); NaN rows stay NaN."""
        if not room:
            return aa
        ok_ = np.isfinite(aa).all(1); out_ = np.full(aa.shape, np.nan)
        if ok_.any():
            out_[ok_] = (R.from_euler("y", yaw_room, degrees=True) * R.from_rotvec(aa[ok_])).as_rotvec()
        return out_
    root_show = to_room(root_fused); root_al_show = {v: to_room(root_al[v]) for v in views}; side = -1.0 if room else 1.0   # the room is watched from the side camera's side
    def quad(pts, rgba):
        m = trimesh.Trimesh(np.asarray(pts, np.float32), np.array([[0, 1, 2], [0, 2, 3]]), process=False); m.visual = trimesh.visual.ColorVisuals(mesh=m, face_colors=np.tile(rgba, (2, 1)))
        return pyrender.Mesh.from_trimesh(m, smooth=False)
    def add_floor_and_mat(sc, floor_y, cx, half_len, cz, big=12.0, mat=True, stripes=False):
        sc.add(quad([[cx - big, floor_y, cz - big], [cx + big, floor_y, cz - big], [cx + big, floor_y, cz + big], [cx - big, floor_y, cz + big]], [205, 200, 190, 255]))
        if mat:
            hl, hw = half_len + mat_pad, mat_half_w
            sc.add(quad([[cx - hl, floor_y + 0.004, cz - hw], [cx + hl, floor_y + 0.004, cz - hw], [cx + hl, floor_y + 0.004, cz + hw], [cx - hl, floor_y + 0.004, cz + hw]], [45, 42, 44, 255]))
            if stripes:                                                        # a cross line every 2 ft, so the walk along the mat shows under a camera that follows the body
                for xs in np.arange(cx - hl, cx + hl + 1e-6, 0.6096):
                    sc.add(quad([[xs - 0.012, floor_y + 0.006, cz - hw], [xs + 0.012, floor_y + 0.006, cz - hw], [xs + 0.012, floor_y + 0.006, cz + hw], [xs - 0.012, floor_y + 0.006, cz + hw]], [120, 116, 118, 255]))
    def look_at(eye, target, up=(0, 1, 0)):
        eye = np.asarray(eye, float); fwd = eye - np.asarray(target, float); fwd /= np.linalg.norm(fwd)
        right = np.cross(np.asarray(up, float), fwd); right /= np.linalg.norm(right); up_ = np.cross(fwd, right)
        Mx = np.eye(4); Mx[:3, 0] = right; Mx[:3, 1] = up_; Mx[:3, 2] = fwd; Mx[:3, 3] = eye; return Mx
    def lights(sc):
        for p_ in ((2, 4, 3), (-2, 3, -2)):
            Lm = np.eye(4); Lm[:3, 3] = p_; sc.add(pyrender.DirectionalLight(intensity=2.5), pose=Lm)
    def finish(rd, sc):
        col, _ = rd.render(sc, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES); col = col.astype(np.float32); al = col[..., 3:4] / 255.0
        return (col[..., :3] * al + 255 * (1 - al)).astype(np.uint8)[:, :, ::-1]
    def body_mesh(Vv, colours, keep_faces=None):
        tm = trimesh.Trimesh(Vv, faces if keep_faces is None else faces[keep_faces], process=False)
        tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=np.c_[colours, np.full(len(Vv), 255)].astype(np.uint8))
        return pyrender.Mesh.from_trimesh(tm, smooth=True)
    view_ref = {"floor": -1.17, "c": np.array([0.0, -0.25, 0.0]), "xz": np.array([xm_, zm])}
    def render(Vv, colours, keep_faces=None, at=None):
        """Close-up of one body (Vv: its vertices posed at the origin; None = the floor only).
        In the truth's room (`room`): the body stands at `at` (room x, z; the last one given when None or NaN) with its lowest
        point on the room's floor, the mat lies where the room has it, and the camera follows the body at a fixed bearing from
        the side camera's side of the room. So the mat and the walking direction agree with the views and the aerial panel.
        Otherwise (no truth, or one without a room): the body at the origin on a bare floor, or on a strip along x."""
        sc = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.45, 0.45, 0.45)); az = np.deg2rad(40)
        if room:
            if at is not None and np.isfinite(at).all():
                view_ref["xz"] = np.asarray(at, float)
            px, pz = view_ref["xz"]
            if Vv is not None:
                sc.add(body_mesh(Vv + np.array([px, -float(Vv[:, 1].min()), pz]), colours, keep_faces))
            add_floor_and_mat(sc, 0.0, xm_, 0.5 * (x1m - x0m), zm, stripes=True)
            c = np.array([px, 0.95, pz]); eye = c + 3.3 * np.array([np.sin(az), 0.25, -np.cos(az)])
        else:
            if Vv is not None:
                sc.add(body_mesh(Vv, colours, keep_faces)); view_ref["floor"] = float(Vv[:, 1].min()) - 0.01; view_ref["c"] = np.array([0.0, Vv[:, 1].mean(), 0.0])
            add_floor_and_mat(sc, view_ref["floor"], 0.0, 2.45, 0.0, mat=G is not None)
            c = view_ref["c"]; eye = c + 3.3 * np.array([np.sin(az), 0.25, np.cos(az)])
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)), pose=look_at(eye, c)); lights(sc); return finish(renderer, sc)
    V_op = V_by = desk = None
    if lay:
        seat = np.zeros(63, np.float32); seat[[0, 3]] = -1.45; seat[[9, 12]] = 1.5
        with torch.no_grad():
            o = bm(global_orient=torch.tensor(R.from_euler("y", lay["operator_yaw_deg"] + 180.0, degrees=True).as_rotvec(), dtype=torch.float32)[None], body_pose=torch.tensor(seat)[None], betas=torch.zeros(1, 10),
                   transl=torch.tensor(np.array(lay["operator_seat_xyz"]) + [0.0, 0.45, 0.0], dtype=torch.float32)[None])
            V_op = o.vertices[0].numpy(); V_op[:, 1] -= V_op[:, 1].min()
            o = bm(global_orient=torch.tensor(R.from_euler("y", 180.0, degrees=True).as_rotvec(), dtype=torch.float32)[None], body_pose=torch.zeros(1, 63), betas=torch.zeros(1, 10),
                   transl=torch.tensor(np.array(lay["bystander_xyz"]) + [0.0, 1.0, 0.0], dtype=torch.float32)[None])
            V_by = o.vertices[0].numpy(); V_by[:, 1] -= V_by[:, 1].min()
        desk = trimesh.creation.box(extents=lay["desk_extents_xyz"]); desk.apply_translation(lay["desk_center_xyz"]); desk.visual = trimesh.visual.ColorVisuals(mesh=desk, face_colors=np.tile([150, 120, 90, 255], (len(desk.faces), 1)))
    trail = []
    def render_aerial(V_world, trail_pts=None):
        sc = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.5, 0.5, 0.5))
        add_floor_and_mat(sc, 0.0, xm_, 0.5 * (x1m - x0m), zm, mat=G is not None)
        if V_world is not None:
            sc.add(body_mesh(V_world, np.tile(COL["truth"] if G is not None else COL["ok"], (len(V_world), 1))))
        if V_op is not None:
            sc.add(body_mesh(V_op, np.tile([120, 120, 160], (len(V_op), 1)))); sc.add(body_mesh(V_by, np.tile([150, 175, 150], (len(V_by), 1)))); sc.add(pyrender.Mesh.from_trimesh(desk, smooth=False))
        for i, (v, cm) in enumerate(cams.items()):
            box = trimesh.creation.box(extents=(0.25, 0.25, 0.25)); box.apply_translation(cm[:3, 3]); box.visual = trimesh.visual.ColorVisuals(mesh=box, face_colors=np.tile(list(VIEW_BGR[views.index(v)][::-1]) + [255], (len(box.faces), 1)))
            sc.add(pyrender.Mesh.from_trimesh(box, smooth=False))
        if trail_pts:
            pts = np.array(trail_pts); pts[:, 1] = 0.02
            sc.add(pyrender.Mesh.from_points(pts, colors=np.tile([60, 160, 60, 255], (len(pts), 1))))
        if G is not None:
            cx = xm_ + 0.6; eye = [cx, 13.0, zm - 2.4]; tgt = [cx, 0.0, zm - 2.2]
        else:                                                      # frame the reconstruction's own path
            h = float(np.clip(1.4 * max(x1m - x0m, 2.0 * float(ext[2])) + 2.5, 5.0, 13.0)); cx = xm_; eye = [cx, h, zm - 0.18 * h]; tgt = [cx, 0.0, zm - 0.17 * h]
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(52)), pose=look_at(eye, tgt, up=(0, 0, -1))); lights(sc); return finish(aerial_r, sc)
    # ---- layout
    PH = 320; vbox = {v: [int(x) for x in K[v]["view_box"]] for v in views}
    pw = {v: int(round(PH * (vbox[v][2] - vbox[v][0]) / (vbox[v][3] - vbox[v][1]))) for v in views}
    n_mid = (3 if G is not None else (2 + (V if V >= 2 else 0)))           # truth | reconstruction | export, or each view's own lift | reconstruction | export
    W = max(sum(pw.values()) + 30 * (V - 1) + 20, n_mid * RW + (n_mid + 1) * 20 + AW + 20); TLH = 330 + (70 if (G is None and V >= 2) else 0)
    light_rect = exp.get("light_rect")
    def label(img, txt, xy, col=(255, 255, 255), scale=0.55, th=1, bg=(0, 0, 0)):
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th); x, y = xy
        cv2.rectangle(img, (x - 3, y - tht - 5), (x + tw + 3, y + 4), bg, -1); cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
    # ---- the static timeline image (matplotlib), cursor drawn per frame
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dpi = 100; fig = plt.figure(figsize=(W / dpi, TLH / dpi), dpi=dpi, facecolor=(0.11, 0.11, 0.11))
    L_, R_ = 0.06, 0.99
    if G is None and V >= 2:
        axs = [fig.add_axes([L_, 0.86, R_ - L_, 0.09]), fig.add_axes([L_, 0.56, R_ - L_, 0.26]), fig.add_axes([L_, 0.27, R_ - L_, 0.26]), fig.add_axes([L_, 0.06, R_ - L_, 0.16])]
    else:
        axs = [fig.add_axes([L_, 0.80, R_ - L_, 0.13]), fig.add_axes([L_, 0.44, R_ - L_, 0.33]), fig.add_axes([L_, 0.08, R_ - L_, 0.33])]
    tt = t_all; win = (tt >= t0) & (tt <= t1)
    dis = None
    if V >= 2:                                                                # where both views vouch for a leg joint: how far apart their rotations are
        a_, b_ = views[0], views[1]; both = usable[a_][:, LEGS] & usable[b_][:, LEGS]
        d_ = geodesic(body[a_][:, LEGS], body[b_][:, LEGS]); d_[~both] = np.nan
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                 # an all-NaN column is a view that saw nothing
            dis = np.nanmedian(d_, axis=1)
    for i, v in enumerate(views):
        y = V - 1 - i
        axs[0].fill_between(tt[win], y - 0.4, y + 0.4, where=leg_ok[v][win], color=VIEW_MPL[i], step="mid", lw=0, alpha=0.9)
    axs[0].set_yticks(list(range(V))); axs[0].set_yticklabels([f"{v} view" for v in views[::-1]], fontsize=7, color="w"); axs[0].set_ylim(-0.6, V - 0.4); axs[0].set_title("usable for the legs", fontsize=8, color="w", loc="left", pad=2)
    for ax, j, name in ((axs[1], 0, "left knee flexion (deg)"), (axs[2], 1, "right knee flexion (deg)")):
        for i, v in enumerate(views):
            ax.plot(tt[win], kn[v][win, j], color=VIEW_MPL[i], lw=0.8, alpha=0.8, label=f"{v} view output")
        if G is not None:
            ax.plot(tt[win], kn["truth"][win, j], color="w", lw=1.0, alpha=0.9, label="truth")
        ax.plot(tt[win], kn["merge"][win, j], color="#3fd13f", lw=2.0, label=f"{tname} ({tfile})")
        ax.set_ylabel(name, fontsize=7, color="w"); ax.set_ylim(0, 90)
    axs[1].legend(fontsize=7, ncol=4, loc="upper left", facecolor=(0.2, 0.2, 0.2), labelcolor="w", framealpha=0.8)
    if len(axs) == 4:
        axs[3].plot(tt[win], dis[win], color="#d0d0d0", lw=0.9); axs[3].fill_between(tt[win], 0, dis[win], color="#d0d0d0", alpha=0.25, lw=0)
        axs[3].set_ylim(0, 45); axs[3].set_ylabel("views disagree (deg)", fontsize=7, color="w")
        axs[3].set_title("leg joints: how far apart the two views' rotations are where both see them (no truth here: this is the available check)", fontsize=7.5, color="w", loc="left", pad=2)
    for ax in axs:
        ax.set_xlim(t0, t1); ax.set_facecolor((0.15, 0.15, 0.15)); ax.tick_params(colors="w", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("0.5")
    axs[0].set_xticklabels([]); axs[1].set_xticklabels([])
    if len(axs) == 4:
        axs[2].set_xticklabels([])
    axs[-1].set_xlabel("IMU time (s)", fontsize=8, color="w")
    tmpdir = Path(tempfile.mkdtemp()); fig.savefig(tmpdir / "timeline.png", dpi=dpi, facecolor=fig.get_facecolor()); plt.close(fig)
    timeline = cv2.imread(str(tmpdir / "timeline.png")); timeline = cv2.resize(timeline, (W, TLH))
    def cursor_x(t): return int(L_ * W + (t - t0) / max(t1 - t0, 1e-6) * (R_ - L_) * W)
    # ---- the filmstrip: every strip_step seconds a small render of what each view outputs, the merge (and the truth)
    ticks = np.arange(t0 + 0.5 * strip_step, t1, strip_step); n_tick = len(ticks); TW = min(56, int((R_ - L_) * W / max(n_tick, 1))); TH = int(TW * 5 / 3); RP = TH + 11
    rows = [(f"{v} view", v) for v in views] + [(tname, None)] + ([("truth", "truth")] if G is not None else []); SH = len(rows) * RP + 20
    thumb_r = _offscreen(pyrender, 2 * TW, 2 * TH)
    def thumb(Vv, colours):
        sc = pyrender.Scene(bg_color=(0.16, 0.16, 0.16, 1.0), ambient_light=(0.5, 0.5, 0.5)); sc.add(body_mesh(Vv, colours))
        c = np.array([0.0, Vv[:, 1].mean(), 0.0]); az = np.deg2rad(40); eye = c + 3.0 * np.array([np.sin(az), 0.2, side * np.cos(az)])
        sc.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(42)), pose=look_at(eye, c)); lights(sc)
        col, _ = thumb_r.render(sc, flags=pyrender.RenderFlags.SKIP_CULL_FACES); return cv2.resize(col[:, :, ::-1], (TW, TH), interpolation=cv2.INTER_AREA)
    strip = np.full((SH, W, 3), 28, np.uint8)
    good = {v: leg_ok[v] & win for v in views}; good[None] = export_ok & win
    row_bgr = {v: VIEW_BGR[i] for i, v in enumerate(views)}; row_bgr[None] = MERGE_BGR
    xL = int(L_ * W); x_of_t = lambda t_: xL + (t_ - t0) / strip_step * TW
    for r_i, (name, key) in enumerate(rows):
        y = 18 + r_i * RP; cv2.putText(strip, name, (4, y + TH // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        for k_i, tk in enumerate(ticks):
            f = int(np.argmin(np.abs(t_all - tk))); x = xL + k_i * TW; img = None
            if key == "truth":
                if f in gi:
                    img = thumb(verts(G["pose_aa"][gi[f], :3], G["pose_aa"][gi[f], 3:66]), np.tile(COL["truth"], (10475, 1)))
            elif key is None:
                if any_lifted[f] and not none[f].all() and np.isfinite(root_fused[f]).all():
                    V_ = verts(root_show[f], merged_draw[f]); cl = np.tile(COL["ok"], (len(V_), 1)); cl[np.isin(dom, np.flatnonzero(none[f]) + 1)] = COL["missing"]; img = thumb(V_, cl)
            elif lifted[key][f] and vis[key][f]:
                V_ = verts(root_al_show[key][f], np.nan_to_num(body[key][f])); cl = np.tile(COL["ok"], (len(V_), 1)); cl[np.isin(dom, np.flatnonzero(~usable[key][f]) + 1)] = COL["missing"]; img = thumb(V_, cl)
            if img is not None:
                if key in good and not good[key][f]:
                    img = (img * 0.45).astype(np.uint8)
                strip[y:y + TH, x:x + TW] = img
            else:
                cv2.rectangle(strip, (x, y), (x + TW - 1, y + TH - 1), (40, 40, 40), -1); cv2.putText(strip, "-", (x + TW // 2 - 4, y + TH // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        if key in good:
            edges = np.flatnonzero(np.diff(np.r_[0, good[key].astype(int), 0]))
            for a_, b_ in zip(edges[::2], edges[1::2]):
                ta, tb = t_all[a_], t_all[b_ - 1]
                cv2.rectangle(strip, (int(x_of_t(ta)), y + TH + 2), (min(int(x_of_t(tb)) + 1, xL + n_tick * TW - 1), y + TH + 5), row_bgr[key], -1)
                ks = [k_ for k_, tk in enumerate(ticks) if ta <= tk <= tb]
                if ks:
                    cv2.rectangle(strip, (xL + ks[0] * TW - 1, y - 2), (xL + (ks[-1] + 1) * TW, y + TH + 7), row_bgr[key], 1)
    cv2.putText(strip, f"every {strip_step:g} s: what each view outputs (red = joints it cannot see), the {tname}" + (", the truth." if G is not None else ".") + "   Bars and boxes mark the well-tracked blocks "
                       "(a view: its legs usable; the merge: the frames the export keeps); thumbnails outside a block are dimmed", (xL, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    thumb_r.delete()
    show_charts, show_strip = readout in ("both", "charts"), readout in ("both", "strips")
    Hh = 20 + PH + 30 + RH + 20 + (TLH if show_charts else 0) + (SH + 10 if show_strip else 0)
    # ---- frames
    import av
    container = av.open(str(out), mode="w"); stream = container.add_stream("libx264", rate=int(round(fps / max(step, 1)))); stream.width, stream.height = W, Hh; stream.pix_fmt = "yuv420p"; stream.options = {"crf": "22"}
    gif_frames = []; g0, g1 = ([float(x) for x in gif_span.split(",")] if gif_span else (t0, min(t1, t0 + 14.0))); err_hist = []
    min_run_s = float(json.loads((rdir / "merge.json").read_text(encoding="utf-8")).get("export", {}).get("min_run_frames", 15)) / fps if (rdir / "merge.json").exists() else 0.5
    try:
        for n, f in enumerate(frames):
            if n % 5 == 0:
                check_stop(stop)
                if progress:
                    progress("demo", n, len(frames), f"{session}/{rid}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
            if not ok:
                break
            canvas = np.full((Hh, W, 3), 28, np.uint8); t = t_all[f]
            x = 10
            for i, v in enumerate(views):                                             # top: the views
                x0, y0, x1, y1 = vbox[v]; crop = img[y0:y1, x0:x1].copy()
                kp = kp_draw[v][f]; legs_merged = bool(collapsed[v][f])
                for a_, b_ in SKELETON:
                    if a_ in FACE_KP or b_ in FACE_KP:
                        continue
                    if np.isfinite(kp[[a_, b_], :2]).all() and kp[a_, 2] >= 0.3 and kp[b_, 2] >= 0.3:
                        bone_col = (40, 190, 250) if (legs_merged and a_ >= 11 and b_ >= 11) else (40, 230, 40)     # amber: both legs found on one leg
                        cv2.line(crop, (int(kp[a_, 0] - x0), int(kp[a_, 1] - y0)), (int(kp[b_, 0] - x0), int(kp[b_, 1] - y0)), bone_col, 3, cv2.LINE_AA)
                crop = cv2.resize(crop, (pw[v], PH), interpolation=cv2.INTER_AREA)
                if legs_merged:
                    label(crop, "2D: both legs on one leg", (8, 22), col=(40, 190, 250), scale=0.42)
                if light_rect and x0 <= light_rect[0] < x1 and y0 <= light_rect[1] < y1:
                    s_ = PH / (y1 - y0); lx, ly = int((light_rect[0] - x0) * s_), int((light_rect[1] - y0) * s_); lw_ = max(int(light_rect[2] * s_), 6)
                    cv2.rectangle(crop, (lx - 3, ly - 3), (lx + lw_ + 3, ly + lw_ + 3), (255, 255, 255), 1); label(crop, "sync light", (lx - 78, ly + lw_ + 2), scale=0.4)
                legs_ok = leg_ok[v][f]; n_us = int(usable[v][f].sum())
                col = (40, 200, 40) if legs_ok else ((30, 170, 240) if lifted[v][f] else (60, 60, 220))
                cv2.rectangle(crop, (0, 0), (pw[v] - 1, PH - 1), col, 4); canvas[20:20 + PH, x:x + pw[v]] = crop
                status = ("subject not tracked here" if not lifted[v][f] else ("bridged frame (not used)" if not vis[v][f] else f"{n_us}/21 usable" + ("" if legs_ok else ", legs at edge") + (" (at rest: box held)" if held[v][f] else "")))
                label(canvas, f"{v} view: {status}" + (" (mirror image: reflected back)" if mirrored[v] else ""), (x + 8, 20 + PH - 10), col=col); x += pw[v] + 30
            yb = 20 + PH + 30; gi_f = gi.get(f); xr = 20; at = truth_xz[f]             # middle: (truth | each view alone), reconstruction, export, aerial
            if G is not None:
                if gi_f is not None:
                    aa = G["pose_aa"][gi_f]; Vt = verts(aa[:3], aa[3:66]); canvas[yb:yb + RH, 20:20 + RW] = render(Vt, np.tile(COL["truth"], (len(Vt), 1)), at=at)
                label(canvas, "rendered truth" + ("  (in its room, on the mat)" if room else ""), (28, yb + RH - 12)); xr = 20 + RW + 20
            elif V >= 2:
                for i, v in enumerate(views):
                    if lifted[v][f] and vis[v][f] and np.isfinite(root_al[v][f]).all():
                        Vv = verts(root_al[v][f], np.nan_to_num(body[v][f])); cv = np.tile(COL["ok"], (len(Vv), 1)); cv[np.isin(dom, np.flatnonzero(~usable[v][f]) + 1)] = COL["missing"]
                        canvas[yb:yb + RH, xr:xr + RW] = render(Vv, cv)
                    else:
                        canvas[yb:yb + RH, xr:xr + RW] = render(None, None)
                        label(canvas, "not tracked here" if not lifted[v][f] else "bridged frame", (xr + 8, yb + RH // 2), col=(60, 60, 220), scale=0.5)
                    label(canvas, f"{v} view alone  ({int(usable[v][f].sum())}/21 usable)", (xr + 8, yb + RH - 12), col=VIEW_BGR[i][::-1] if False else (255, 255, 255))
                    cv2.rectangle(canvas, (xr, yb), (xr + RW - 1, yb + RH - 1), VIEW_BGR[i], 2); xr += RW + 20
            if not any_lifted[f]:
                label(canvas, "no data: the subject is tracked in no view", (xr + 8, yb + RH // 2), col=(60, 60, 220), scale=0.5)
            elif none[f].all() or not np.isfinite(root_fused[f]).all():
                label(canvas, "no usable joint in any view", (xr + 8, yb + RH // 2), col=(60, 60, 220))
            else:
                Vr = verts(root_show[f], merged_draw[f]); colours = np.tile(COL["ok"], (len(Vr), 1)); colours[np.isin(dom, np.flatnonzero(none[f]) + 1)] = COL["missing"]
                canvas[yb:yb + RH, xr:xr + RW] = render(Vr, colours, at=at)
                if gi_f is not None:
                    e = geodesic(merged_out[f], G["pose_aa"][gi_f, 3:66].reshape(21, 3)); le = np.nanmedian(e[LEGS]) if np.isfinite(e[LEGS]).any() else np.nan
                    if np.isfinite(le):
                        err_hist.append(le)
                    label(canvas, f"leg-joint error vs truth: {le:5.1f} deg" if np.isfinite(le) else "leg joints: no usable view", (xr + 8, yb + 24))
                    if room:
                        he = float(geodesic(root_show[f][None], truth_root[f][None])[0])
                        label(canvas, f"root orientation error: {he:5.1f} deg" if np.isfinite(he) else "root orientation: none", (xr + 8, yb + 46), scale=0.5)
                        label(canvas, f"in the room: one yaw for the recording ({yaw_room:+.0f} deg), true position", (xr + 8, yb + 64), scale=0.36, col=(200, 200, 200))
                src = []
                for grp, nm in ((LEGS, "legs"), (ARMS, "arms"), (SPINE, "torso")):
                    who = [v for v in views if usable[v][f][grp].any()]
                    src.append(f"{nm}: " + ("both" if len(who) == V and V > 1 else (who[0] if len(who) == 1 else ("all" if who else "none"))))
                label(canvas, ("reconstruction  " if target == "merge" else "joint two-view fit  ") + " | ".join(src), (xr + 8, yb + RH - 12), scale=0.5)
            label(canvas, "red = no view could see this part (not output)", (xr + 8, yb + RH - 34), col=(90, 90, 230), scale=0.5)
            label(canvas, "2D skeletons: ViTPose keypoints (the lifter's input); amber legs = the detector put both legs on one leg (the 3D lifter copes, measured)",
                  (10, 20 + PH + 8), scale=0.4, col=(200, 200, 200), bg=(28, 28, 28))
            xe = xr + RW + 20                                                           # the export
            if export_ok[f] and np.isfinite(root_fused[f]).all():
                Ve = verts(root_show[f], np.nan_to_num(merged_out[f])); keep = ~np.isin(dom[faces], np.flatnonzero(none[f]) + 1).any(1)
                canvas[yb:yb + RH, xe:xe + RW] = render(Ve, np.tile(COL["ok"], (len(Ve), 1)), keep, at=at)
            else:
                canvas[yb:yb + RH, xe:xe + RW] = render(None, None, at=at)
                why = ("subject tracked in no view" if not any_lifted[f] else ("a leg joint seen by no view" if not np.isfinite(merged_out[f][LEGS]).all()
                       else ("a torso joint seen by no view" if not np.isfinite(merged_out[f][SPINE]).all() else f"run shorter than {min_run_s:.1f} s")))
                label(canvas, "frame not exported:", (xe + 8, yb + RH // 2 - 22), col=(60, 60, 220), scale=0.5); label(canvas, why, (xe + 8, yb + RH // 2), col=(60, 60, 220), scale=0.5)
            label(canvas, f"frames kept so far: {export_ok[win & (tt <= t)].mean():.0%}", (xe + 8, yb + RH - 56), scale=0.5)
            label(canvas, "an arm no view saw is absent; legs and torso are required", (xe + 8, yb + RH - 34), scale=0.42)
            label(canvas, f"export  ({tfile}: what a user takes away)", (xe + 8, yb + RH - 12))
            xt = xe + RW + 20
            if G is not None:
                Vw = verts(G["pose_aa"][gi_f, :3], G["pose_aa"][gi_f, 3:66], G["transl"][gi_f]) if gi_f is not None else None
                canvas[yb:yb + AH, xt:xt + AW] = render_aerial(Vw)
                label(canvas, "aerial truth  (" + ", ".join(f"{v} camera" for v in cams) + ")", (xt + 6, yb + AH - 8), scale=0.42)
            else:
                Vw = None
                if export_ok[f] and np.isfinite(transl_f[f]).all() and np.isfinite(root_fused[f]).all():
                    Vw = verts(root_fused[f], np.nan_to_num(merged_out[f]), transl_f[f]); trail.append(transl_f[f].copy())
                canvas[yb:yb + AH, xt:xt + AW] = render_aerial(Vw, trail[-600:])
                label(canvas, "aerial: the reconstruction on its own path (reference view's world)", (xt + 6, yb + AH - 8), scale=0.42)
            lines = [f"IMU t = {t:6.2f} s   (video frame {f})", "3D reconstruction from partial views:", "a joint is output only from the view(s) that",
                     "see it, cross-faded over 0.5 s at segment edges;", "nothing where no view sees it.",
                     "legs usable so far: " + "  ".join(f"{v} {leg_ok[v][win & (tt <= t)].mean():.0%}" for v in views) + f"  any {np.any([leg_ok[v] for v in views], axis=0)[win & (tt <= t)].mean():.0%}"]
            if err_hist:
                lines += [f"median leg error so far: {np.nanmedian(err_hist):.1f} deg"]
            if slope:
                lines += ["lifted floor slope (0 = level): " + "  ".join(f"{v} {s_:.1f}" for v, s_ in slope.items()) + " deg"]
            for k_, ln in enumerate(lines):
                cv2.putText(canvas, ln, (xt, yb + AH + 26 + 22 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
            yt = yb + RH + 20; cx = cursor_x(t)
            if show_strip:
                canvas[yt:yt + SH] = strip
                k_cur = int(np.clip((t - t0) // strip_step, 0, max(n_tick - 1, 0))); x_c = int(L_ * W) + k_cur * TW
                cv2.rectangle(canvas, (x_c - 2, yt + 14), (x_c + TW + 1, yt + SH - 2), (255, 255, 255), 1); yt += SH + 10
            if show_charts:
                canvas[yt:yt + TLH] = timeline; cv2.line(canvas, (cx, yt + 8), (cx, yt + TLH - 24), (255, 255, 255), 2)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(canvas[:, :, ::-1].copy(), format="rgb24")):
                container.mux(pkt)
            if gif and g0 <= t <= g1 and n % 3 == 0:
                gif_frames.append(cv2.resize(canvas, (W // 2, Hh // 2), interpolation=cv2.INTER_AREA))
            if n % 300 == 0:
                log(f"  {n}/{len(frames)} frames", flush=True) if log is print else log(f"  {n}/{len(frames)} frames")
    except RunStopped:                                                           # nothing half-written is left behind
        container.close(); cap.release()
        out.unlink(missing_ok=True); raise
    for pkt in stream.encode():
        container.mux(pkt)
    container.close(); renderer.delete(); aerial_r.delete(); cap.release()
    res = {"video": str(out), "n_frames": len(frames), "size": [W, Hh], "session": session, "recording": rid}
    log(f"demo -> {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames, {W}x{Hh})")
    if gif and gif_frames:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        for i, fr_ in enumerate(gif_frames):
            cv2.imwrite(str(tmpdir / f"g{i:04d}.png"), fr_)
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmpdir / "g%04d.png"), "-vf", "palettegen=max_colors=128", str(tmpdir / "pal.png")], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(tmpdir / "g%04d.png"), "-i", str(tmpdir / "pal.png"), "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", gif], check=True)
        res["gif"] = gif; log(f"gif -> {gif} ({Path(gif).stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames, IMU {g0}-{g1} s)")
    return res
