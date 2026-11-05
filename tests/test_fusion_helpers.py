"""Small helpers of the fusion library: the mirror map (exact for the SMPL-X template by forward kinematics) and the
collapsed-legs test of a 2D skeleton."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wgvp.fusion import L_FOOT, R_FOOT, collapsed_legs, fk_joints, mirror_body, mirror_check, mirror_root, mirror_transl   # noqa: E402


def test_mirror_map_reflects_the_body():
    rng = np.random.default_rng(0); T = 200
    body = rng.normal(0, 0.4, (T, 21, 3)); root = R.from_euler("yx", np.c_[rng.uniform(-180, 180, T), rng.normal(0, 5, T)], degrees=True).as_rotvec(); tr = rng.normal(0, 1, (T, 3))
    J = fk_joints(root, body, tr); Jm = fk_joints(mirror_root(root), mirror_body(body), mirror_transl(tr))
    assert np.abs(Jm[:, L_FOOT] - J[:, R_FOOT] * [-1, 1, 1]).max() < 0.05 and np.abs(Jm[:, R_FOOT] - J[:, L_FOOT] * [-1, 1, 1]).max() < 0.05   # template asymmetry: mm
    assert np.abs(mirror_body(mirror_body(body)) - body).max() == 0.0
    chk = mirror_check(body, mirror_body(body), np.ones(T, bool)); assert chk["mirrored"] and chk["mirrored_deg"] < 1e-6 < chk["as_is_deg"]
    assert not mirror_check(body, body, np.ones(T, bool))["mirrored"] and not mirror_check(body, mirror_body(body), np.arange(T) < 20)["mirrored"]   # too few frames: no verdict


def test_collapsed_legs_flags_both_legs_on_one_leg():
    kp = np.zeros((4, 17, 3)); kp[:, :, 2] = 0.9
    kp[:, 13] = [100, 200, 0.9]; kp[:, 15] = [100, 300, 0.9]                       # left knee / ankle
    kp[0, 14] = [160, 200, 0.9]; kp[0, 16] = [170, 300, 0.9]                        # frame 0: right leg apart
    kp[1, 14] = [101, 201, 0.9]; kp[1, 16] = [100, 301, 0.9]                        # frame 1: both legs on one leg
    kp[2, 14] = [160, 200, 0.9]; kp[2, 16] = [101, 300, 0.9]                        # frame 2: ankles together, knees apart (a crossing)
    kp[3, 14] = [101, 201, 0.1]; kp[3, 16] = [100, 301, 0.9]                        # frame 3: collapsed but not confident
    assert collapsed_legs(kp).tolist() == [False, True, False, False]
