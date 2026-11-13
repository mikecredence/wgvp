"""The joint two-view fit on fabricated data: one true motion seen by two cameras of known pose. Each view's lift
has the faults a monocular lift has (a depth error along its line of sight, a wrong camera pitch against gravity),
the merge it starts from has a drifting pelvis path and biased knees, and the 2D keypoints are true. The fit must
find the cameras' relative pose, put the path and the knees right, take the up direction from the two cameras'
level axes instead of either lift's gravity, and write every documented field. The SMPL-X model file is not in the
repository, so the tests run on the mean skeleton (no shape directions)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.cli import main as cli_main, merged_sessions   # noqa: E402
from wgvp.fusion import geodesic   # noqa: E402
from wgvp.jointfit import (KPJ, KPS, KP_MIRROR, body_joints, jointfit_recording, kabsch, level_axis, lift_up, load_skeleton, load_views,   # noqa: E402
                           mean_skeleton, unmirror_view, up_from_levels)

FPS, F, SIZE = 30.0, 700.0, (900.0, 600.0)
BOX = {"left": ([0, 0, 900, 600], 1.0), "right": ([900, 0, 2700, 1200], 0.5)}                 # the right panel is cut at half scale
CAM = {"left": ([0.3, 1.5, 9.5], 6.0, 1.06, 0.0), "right": ([-5.5, 2.6, 2.0], -9.0, 0.95, 6.0)}   # centre; the lift's errors: camera pitch (deg), depth scale, body yaw against the camera (deg)
TARGET = np.array([0.0, 0.9, 2.0]); K = np.array([[F, 0, SIZE[0] / 2], [0, F, SIZE[1] / 2], [0, 0, 1.0]]); FAST = {"iters": (40, 160)}


def _camera(name):
    """World -> camera rotation (x right, y down, z ahead; no roll) and the camera's centre."""
    c = np.array(CAM[name][0]); z = TARGET - c; z /= np.linalg.norm(z); x = np.cross([0.0, -1.0, 0.0], z); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z]), c


def _truth(T, rng):
    t = np.arange(T) / FPS; body = np.zeros((T, 21, 3)); s = np.sin(2 * np.pi * 0.9 * t)
    body[:, 0, 0] = -0.35 * s; body[:, 1, 0] = 0.35 * s; body[:, 3, 0] = 0.5 + 0.4 * s; body[:, 4, 0] = 0.5 - 0.4 * s          # hips and knees
    body[:, 15, 0] = 0.3 * s; body[:, 16, 0] = -0.3 * s; body[:, 15, 2] = -1.1; body[:, 16, 2] = 1.1                            # shoulders: arms down, swinging
    body[:, 17, 1] = -0.4; body[:, 18, 1] = 0.4                                                                               # elbows a little bent
    body += np.cumsum(rng.normal(0, 0.003, (T, 21, 3)), axis=0)
    heading = 8.0 * np.sin(2 * np.pi * 0.1 * t)
    root = np.stack([(R.from_euler("y", h, degrees=True) * R.from_euler("x", 5, degrees=True)).as_rotvec() for h in heading])
    fwd = R.from_euler("y", heading, degrees=True).apply([0.0, 0.0, 1.0]); transl = np.cumsum(1.2 * fwd / FPS, axis=0) + [0.0, 0.95, 0.0]
    return body, root, transl


def _view_dict(name, body, root, transl, skel, rng, visible=None):
    """One view as `load_views` returns it: true keypoints, and a lift with a depth error, a wrong pitch against gravity and (the right view) the body turned against the camera."""
    Rc, c = _camera(name); _, pitch, depth, yaw = CAM[name]; T = len(root); J = body_joints(root, body, transl, np.zeros(10), skel)
    Xc = (J - c) @ Rc.T; kp = np.zeros((T, 17, 3)); p = Xc[:, KPJ] @ K.T; kp[:, KPS, :2] = p[:, :, :2] / p[:, :, 2:]; kp[:, KPS, 2] = 0.9
    cam_root = (R.from_euler("y", yaw, degrees=True) * R.from_matrix(Rc) * R.from_rotvec(root)).as_rotvec()
    glob_root = (R.from_euler("y", 40.0 if name == "left" else -70.0, degrees=True) * R.from_matrix(Rc.T) * R.from_euler("x", pitch, degrees=True) * R.from_rotvec(cam_root)).as_rotvec()
    cam_body = (R.from_rotvec(body.reshape(-1, 3)) * R.from_rotvec(rng.normal(0, 0.04, (T * 21, 3)))).as_rotvec().reshape(T, 21, 3)
    return {"name": name, "kp": kp, "K": K.copy(), "size": SIZE, "detected": np.ones(T, bool) if visible is None else visible, "cam_root": cam_root, "cam_body": cam_body,
            "cam_transl": depth * Xc[:, 0] - skel[0][0], "glob_root": glob_root, "betas": np.zeros(10), "usable": np.ones((T, 21), bool), "mirrored": False}


def _write_view(d: Path, v, frames):
    d.mkdir(parents=True); T = len(frames); box, sc = BOX[v["name"]]; kp = v["kp"].copy(); kp[:, :, 0] = kp[:, :, 0] / sc + box[0]; kp[:, :, 1] = kp[:, :, 1] / sc + box[1]
    np.savez(d / "smpl_global.npz", frame_idx=frames, imu_t_s=frames / FPS - 2.0, body_pose=v["cam_body"].reshape(T, 63).astype(np.float32), global_orient=v["glob_root"],
             transl=np.zeros((T, 3), np.float32), betas=np.zeros((T, 10), np.float32), visible=v["detected"])
    np.savez(d / "smpl_incam.npz", frame_idx=frames, body_pose=v["cam_body"].reshape(T, 63), global_orient=v["cam_root"], transl=v["cam_transl"], betas=np.zeros((T, 10)))
    np.savez(d / "kp2d_vitpose.npz", kp2d=kp.astype(np.float32), view_box=np.array(box), cut_scale=sc, K_fullimg_cut=v["K"], frame_idx=frames)


def _recording(d: Path, T=90, seed=0, reference="left", mirrored=(), views=("left", "right"), visible=None):
    """A fabricated merged recording with its view folders. Returns the truth."""
    rng = np.random.default_rng(seed); skel = mean_skeleton(); body, root, transl = _truth(T, rng); frames = np.arange(100, 100 + T)
    for name in views:
        v = _view_dict(name, body, root, transl, skel, rng, None if visible is None else visible.get(name))
        _write_view(d / name, unmirror_view(v) if name in mirrored else v, frames)
    mb = body.copy(); mb[:, 3, 0] += 0.15; mb[:, 4, 0] -= 0.10; mb[:10, 18] = np.nan                                           # biased knees; an elbow the merge does not output
    drift = 0.6 * np.sin(np.pi * np.arange(T) / T)[:, None] * np.array([0.6, 0.0, 0.8]); ex = np.ones(T, bool); ex[:3] = False
    np.savez(d / "merged_smpl.npz", views=np.array(views), frame_idx=frames, imu_t_s=frames / FPS - 2.0, body_pose=mb.reshape(T, 63).astype(np.float32), global_orient=root.astype(np.float32),
             transl=(transl + drift).astype(np.float32), export_ok=ex, usable=np.ones((len(views), T, 21), bool), reference_view=reference,
             view_mirrored=np.array([n in mirrored for n in views]), fps=FPS)
    return {"body": body, "root": root, "transl": transl, "merged_body": mb, "merged_transl": transl + drift, "export_ok": ex}


def _path_err(est, true, ok):
    d = (est - true)[ok][:, [0, 2]]; return float(np.median(np.linalg.norm(d - np.median(d, axis=0), axis=1)))


def test_level_axes_give_the_up_a_wrong_pitch_cannot_move():
    (Ra, _), (Rb, _) = _camera("left"), _camera("right"); up_a = Ra @ [0.0, 1.0, 0.0]; Rab = Rb @ Ra.T
    tilt = lambda Rc, deg: R.from_euler("x", deg, degrees=True).inv().apply(Rc @ [0.0, 1.0, 0.0])                             # a lift that has the camera's pitch wrong
    ha, hb = level_axis(tilt(Ra, 8.0)), level_axis(tilt(Rb, -11.0)); assert np.allclose(ha, level_axis(up_a)) and abs(ha[2]) < 1e-12
    up, from_levels = up_from_levels(ha, Rab.T @ hb, tilt(Ra, 8.0)); assert from_levels
    assert np.degrees(np.arccos(np.clip(up @ up_a, -1, 1))) < 1e-6 and np.degrees(np.arccos(tilt(Ra, 8.0) @ up_a)) > 7.9      # the cross product is right where the lift's own up is 8 deg off
    hint = tilt(Ra, 8.0); up, from_levels = up_from_levels(ha, R.from_euler("y", 5, degrees=True).apply(ha), hint)            # two cameras on one line: no cross product to trust
    assert not from_levels and np.allclose(up, hint / np.linalg.norm(hint))


def test_kabsch_and_lift_up():
    rng = np.random.default_rng(1); P = rng.normal(size=(40, 3)); Rt = R.from_rotvec([0.3, -1.1, 0.2]).as_matrix(); t = np.array([0.5, -2.0, 4.0])
    Rm, tm = kabsch(P, P @ Rt.T + t); assert np.allclose(Rm, Rt, atol=1e-9) and np.allclose(tm, t, atol=1e-9)
    Rc, _ = _camera("right"); root = R.from_euler("y", np.linspace(0, 50, 20), degrees=True).as_rotvec()
    cam = (R.from_matrix(Rc) * R.from_rotvec(root)).as_rotvec(); glob = (R.from_euler("y", 33, degrees=True) * R.from_rotvec(root)).as_rotvec()
    assert np.allclose(lift_up(glob, cam, np.ones(20, bool)), Rc @ [0.0, 1.0, 0.0], atol=1e-9)                                # a view's world may be turned about the vertical: its up is not


def test_unmirror_is_an_involution_and_matches_the_mirrored_picture():
    rng = np.random.default_rng(2); J0, Jd = mean_skeleton(); Js = J0.copy()
    for l, r in ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)):                                    # a left/right symmetric skeleton, so a mirrored pose is a mirrored body
        m = 0.5 * (J0[l] * [-1, 1, 1] + J0[r]); Js[r] = m; Js[l] = m * [-1, 1, 1]
    Js[[0, 3, 6, 9, 12, 15], 0] = 0.0; skel = (Js, Jd); body, root, transl = _truth(20, rng); v = _view_dict("left", body, root, transl, skel, rng)
    v["cam_body"] = body; v["cam_transl"] = v["cam_transl"] / CAM["left"][2]                                                 # an exact lift: it must reproject onto its keypoints
    w = unmirror_view(v); u = unmirror_view(w)
    for k in ("kp", "K", "cam_root", "cam_body", "cam_transl", "glob_root"):
        assert np.allclose(u[k], v[k], atol=1e-9), k
    assert np.allclose(w["kp"][:, 5, 0], SIZE[0] - v["kp"][:, 6, 0]) and KP_MIRROR[5] == 6
    for x in (v, w):                                                                                                          # the mirrored lift is the body the mirrored picture shows
        p = body_joints(x["cam_root"], x["cam_body"], x["cam_transl"], np.zeros(10), skel)[:, KPJ] @ x["K"].T
        assert np.abs(p[:, :, :2] / p[:, :, 2:] - x["kp"][:, KPS, :2]).max() < 1e-6


def test_joint_fit_recovers_cameras_path_and_knees(tmp_path):
    d = tmp_path / "Rec01"; tr = _recording(d); logs = []
    s = jointfit_recording(d, mean_skeleton(), log=logs.append); z = np.load(d / "jointfit_smpl.npz"); j = json.loads((d / "jointfit.json").read_text())      # the shipped settings
    assert s["status"] == "PASS" and j["status"] == "PASS" and j["views"] == ["left", "right"] and not j["cameras"]["given"], j["flags"]
    T = len(tr["root"]); ex = tr["export_ok"]; body = z["body_pose"].reshape(T, 21, 3)
    for k in ("frame_idx", "imu_t_s", "body_pose", "global_orient", "transl", "betas", "export_ok", "n_keypoints", "seen", "seen_both", "reproj_px", "cam_b_from_a_R", "cam_b_from_a_t",
              "world_from_cam_a_R", "keypoint_offsets_m", "views", "fps", "fields_doc", "note"):
        assert k in z.files, k
        if k not in ("fps", "note", "fields_doc"):
            assert k in str(z["fields_doc"]), k
    assert z["body_pose"].shape == (T, 63) and z["n_keypoints"].shape == (2, T) and z["reproj_px"].shape == (2, T) and np.array_equal(z["export_ok"], ex)
    assert np.isnan(body[:10, 18]).all() and np.isfinite(body[10:]).all() and np.isfinite(z["transl"]).all()                  # NaN where the merge is, nowhere else
    assert (z["n_keypoints"] == 12).all() and z["seen_both"].all() and j["views_complete_frac"] == 1.0 and j["frames_seen_by_one"] == 0
    assert all(j["reprojection_px"][v]["median"] < 1.5 and j["reprojection_px"][v]["set_aside_frac"] < 0.02 for v in ("left", "right")), j["reprojection_px"]
    # the cameras: camera b against camera a, found from the walker. The seed carries the two lifts' depth errors; the scale of a
    # self-calibrated scene is the body's size as the lifts saw it, so the baseline's length is only held to a few percent
    (Ra, ca), (Rb, cb) = _camera("left"), _camera("right"); Rt, tt = Rb @ Ra.T, Rb @ (ca - cb); t = z["cam_b_from_a_t"]
    ang = lambda A, B: np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))); between = lambda u, v: np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    assert ang(z["cam_b_from_a_R"], Rt) < 1.5 and between(t, tt) < 1.5 and abs(np.linalg.norm(t) / np.linalg.norm(tt) - 1) < 0.08, (ang(z["cam_b_from_a_R"], Rt), between(t, tt), np.linalg.norm(t) / np.linalg.norm(tt))
    assert j["cameras"]["moved_from_seed_m"] > 0.1 and abs(j["cameras"]["baseline_m"] - np.linalg.norm(t)) < 1e-9
    # the up: both lifts have their camera's pitch wrong (6 and 9 deg), the two level axes do not
    up_a = Ra @ [0.0, 1.0, 0.0]; assert j["up"]["rule"].startswith("cross product") and np.degrees(np.arccos(np.clip(z["world_from_cam_a_R"][1] @ up_a, -1, 1))) < 1.0
    # the path and the knees: better than the merge they started from
    e_fit, e_merge = _path_err(z["transl"], tr["transl"], ex), _path_err(tr["merged_transl"], tr["transl"], ex); assert e_merge > 0.10 and e_fit < 0.07, (e_fit, e_merge)
    k_fit, k_merge = np.median(geodesic(body, tr["body"])[ex][:, [3, 4]]), np.median(geodesic(tr["merged_body"], tr["body"])[ex][:, [3, 4]]); assert k_merge > 6.0 and k_fit < 3.5, (k_fit, k_merge)
    assert np.median(geodesic(z["global_orient"], tr["root"])[ex]) < 1.0 and j["change_from_merge"]["path_m_median"] > 0.05 and j["change_from_merge"]["legs_deg_median"] > 0.5
    assert any("joint fit: PASS" in l for l in logs)


def test_the_merges_reference_view_hosts_the_fit_and_known_cameras_are_held(tmp_path):
    d = tmp_path / "Rec02"; tr = _recording(d, reference="right"); (Ra, ca), (Rb, cb) = _camera("right"), _camera("left"); Rt, tt = Rb @ Ra.T, Rb @ (ca - cb); ex = tr["export_ok"]
    s = jointfit_recording(d, mean_skeleton(), params=FAST, log=lambda *_: None, write=False, return_arrays=True)
    assert s["views"] == ["right", "left"] and not (d / "jointfit_smpl.npz").exists() and not (d / "jointfit.json").exists()   # write=False leaves the folder alone
    assert np.degrees(np.linalg.norm(R.from_matrix(s["arrays"]["cam_b_from_a_R"] @ Rt.T).as_rotvec())) < 4.0                    # camera b is now the left view
    g = jointfit_recording(d, mean_skeleton(), params=FAST, cams=(Rt, tt), log=lambda *_: None, write=False, return_arrays=True)
    assert g["cameras"]["given"] and np.allclose(g["arrays"]["cam_b_from_a_R"], Rt) and np.allclose(g["arrays"]["cam_b_from_a_t"], tt)
    assert _path_err(g["arrays"]["transl"], tr["transl"], ex) < 0.03                                                          # a known baseline fixes the scale the walker cannot


def test_the_output_is_smoothed_inside_runs_only(tmp_path):
    d = tmp_path / "Rec06"; _recording(d); quiet = lambda *_: None; acc = lambda x: np.nanmean(np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1))
    raw = jointfit_recording(d, mean_skeleton(), params={**FAST, "smooth_frames": 1}, log=quiet, write=False, return_arrays=True)["arrays"]
    out = jointfit_recording(d, mean_skeleton(), params=FAST, log=quiet, write=False, return_arrays=True)["arrays"]
    assert acc(out["transl"]) < 0.8 * acc(raw["transl"]) and acc(out["body_pose"][:, 9:12]) < 0.8 * acc(raw["body_pose"][:, 9:12])   # the pelvis path and a knee
    for k in ("body_pose", "global_orient", "transl"):
        assert (np.isfinite(out[k]) == np.isfinite(raw[k])).all(), k                                                          # gaps stay gaps, nothing is filled across them


def test_a_mirrored_view_is_reflected_back(tmp_path):
    d0, d1 = tmp_path / "plain" / "Rec03", tmp_path / "mirrored" / "Rec03"; _recording(d0); _recording(d1, mirrored=("right",)); m = np.load(d1 / "merged_smpl.npz")
    a, b = load_views(d0, np.load(d0 / "merged_smpl.npz")), load_views(d1, m); assert b[1]["mirrored"] and not b[0]["mirrored"]
    for k in ("kp", "K", "cam_root", "cam_body", "cam_transl", "glob_root"):
        assert np.allclose(a[1][k], b[1][k], atol=1e-4), k                                                                    # float32 keypoints on disk
    s = jointfit_recording(d1, mean_skeleton(), params=FAST, log=lambda *_: None)
    assert s["status"] == "WARN" and any("mirrored" in f for f in s["flags"]) and s["reprojection_px"]["right"]["median"] < 1.5


def test_two_panels_of_one_camera_are_flagged_as_one_viewpoint(tmp_path):
    d = tmp_path / "twin" / "Rec05"; _recording(d); shutil.rmtree(d / "right"); shutil.copytree(d / "left", d / "right")       # the second "view" is the first camera again
    s = jointfit_recording(d, mean_skeleton(), params=FAST, log=lambda *_: None)
    assert s["status"] == "WARN" and s["cameras"]["baseline_m"] < 0.05 and any("one viewpoint" in f for f in s["flags"])
    d = tmp_path / "pair" / "Rec05"; _recording(d); s = jointfit_recording(d, mean_skeleton(), params=FAST, log=lambda *_: None)
    assert s["cameras"]["baseline_m"] > 1.0 and not any("one viewpoint" in f for f in s["flags"])


def test_recordings_the_fit_cannot_take_are_skipped(tmp_path):
    skel = mean_skeleton(); quiet = lambda *_: None
    assert jointfit_recording(tmp_path / "nothing", skel, log=quiet) is None
    d = tmp_path / "one" / "Rec04"; _recording(d, views=("left",)); s = jointfit_recording(d, skel, log=quiet)
    assert s["status"] == "SKIP" and "two views" in s["flags"][0] and not (d / "jointfit_smpl.npz").exists() and json.loads((d / "jointfit.json").read_text())["status"] == "SKIP"
    d = tmp_path / "noincam" / "Rec04"; _recording(d); (d / "right" / "smpl_incam.npz").unlink()
    assert jointfit_recording(d, skel, log=quiet)["status"] == "SKIP"
    T = 90; seen = {"left": np.arange(T) < 40, "right": np.arange(T) >= 50}; d = tmp_path / "apart" / "Rec04"; _recording(d, visible=seen)   # the views never show the subject together
    s = jointfit_recording(d, skel, params=FAST, log=quiet); assert s["status"] == "SKIP" and "never see the subject together" in s["flags"][0]


def test_cli_needs_the_model_file(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        load_skeleton(tmp_path / "no_models")
    d = tmp_path / "wgvp_output" / "Sub 001" / "deliver" / "Rec05"; _recording(d)
    assert cli_main(["jointfit", str(tmp_path), "--body-models", str(tmp_path / "no_models")]) == 1 and "SMPLX_NEUTRAL" in capsys.readouterr().out
    assert not (d / "jointfit_smpl.npz").exists()


def test_the_demo_finds_its_sessions_without_being_told(tmp_path):
    """`wgvp demo <root>` with no --session renders every session that holds a merged recording: the session is the
    folder two above the recording's, not the `deliver` folder between them."""
    for sid, rec in (("Sub 001", "Sub001_selfpace"), ("Sub 001", "Sub001_tug"), ("Sub 002", "Sub002_selfpace")):
        d = tmp_path / "wgvp_output" / sid / "deliver" / rec; d.mkdir(parents=True); (d / "merged_smpl.npz").write_bytes(b"")
    (tmp_path / "wgvp_output" / "Sub 003" / "deliver" / "Sub003_selfpace").mkdir(parents=True)      # lifted, not merged
    assert merged_sessions(tmp_path) == ["Sub 001", "Sub 002"] and merged_sessions(tmp_path / "nothing") == []
