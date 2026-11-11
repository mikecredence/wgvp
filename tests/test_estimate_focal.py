"""tools/estimate_focal.py on the segments of a room drawn by a camera whose focal length is known."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.estimate_focal import DIAG_35, manhattan_focal

W, H = 1280, 720


def room_segments(focal_35mm, yaw_deg, pitch_deg, n=40, noise_px=0.3, seed=0):
    """Segments (pixels) of straight edges along the three axes of a room, seen by a camera at its centre."""
    rng = np.random.default_rng(seed)
    f = focal_35mm * np.hypot(W, H) / DIAG_35
    R = Rotation.from_euler("xy", [pitch_deg, yaw_deg], degrees=True).as_matrix()        # room to camera
    S = []
    for axis in range(3):
        while sum(1 for s in S if s[4] == axis) < n:
            p = rng.uniform(-3, 3, 3); p[2] = rng.uniform(4, 12)                         # a point in front, camera frame
            a, b = p, p + R[:, axis] * rng.uniform(0.8, 2.5)
            if min(a[2], b[2]) < 1:
                continue
            uv = np.r_[f * a[:2] / a[2], f * b[:2] / b[2]] + np.r_[W / 2, H / 2, W / 2, H / 2]
            if (uv[[0, 2]] < 0).any() or (uv[[0, 2]] > W).any() or (uv[[1, 3]] < 0).any() or (uv[[1, 3]] > H).any():
                continue
            if np.hypot(uv[2] - uv[0], uv[3] - uv[1]) < 40:
                continue
            S.append(np.r_[uv + rng.normal(0, noise_px, 4), axis])
    return np.array(S)[:, :4]


@pytest.mark.parametrize("focal,yaw,pitch", [(17.0, 25, 12), (24.0, -35, 8), (35.0, 40, -10)])
def test_known_camera_is_recovered(focal, yaw, pitch):
    r = manhattan_focal(room_segments(focal, yaw, pitch), (W, H))
    assert r is not None and not r["weak"]
    assert r["focal_35mm"] == pytest.approx(focal, rel=0.03)
    assert r["pairs_35mm"] and all(p == pytest.approx(focal, rel=0.06) for p in r["pairs_35mm"])


def test_camera_square_to_the_wall_is_refused():
    """Two of the three families are parallel in the picture: any focal length explains them, so none is given."""
    r = manhattan_focal(room_segments(24.0, 0, 0), (W, H))
    assert r is None or r["weak"]


def test_one_direction_alone_is_refused():
    S = room_segments(24.0, 25, 12)[:40]            # the first axis only
    r = manhattan_focal(S, (W, H))
    assert r is None or r["weak"]
