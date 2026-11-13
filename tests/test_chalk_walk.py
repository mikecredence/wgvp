"""tools/chalk_walk.py on a walker drawn by a pinhole camera over ground that rises along the walk."""
import numpy as np

from tools.chalk_walk import BOX_OVER_HEIGHT, find_turns, ground_path, height_reading, score

FPS, F, C = 30.0, 1300.0, np.array([960.0, 540.0])
CAM_H, STATURE, LINES = 0.8, 1.7, (-2.0, 1.9)


def scene(rise_deg, yaw_deg=0.0, seconds=30):
    """Marks (4, 2), pelvis column, feet row, box height and the walker's true position along the ground."""
    s, a = np.radians(rise_deg), np.radians(yaw_deg); t = np.arange(int(seconds * FPS)) / FPS
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])     # world to camera, y down

    def ground(x, z):          # a point x along the walk and z away from the camera, on ground that rises with x
        return np.stack([x * np.cos(s), CAM_H - x * np.sin(s), z + 0 * x], -1)

    def px(P):
        P = P @ R.T; return C + F * P[..., :2] / P[..., 2:3]

    x = 2.15 * np.sin(2 * np.pi * t / 10) - 0.05; foot = ground(x, 6.0 + 0.1 * np.sin(t)); head = foot - [0, STATURE, 0]
    marks = np.concatenate([px(ground(np.full(2, v), np.array([5.2, 6.8]))) for v in LINES])
    return marks, px(foot)[:, 0], px(foot)[:, 1], px(foot)[:, 1] - px(head)[:, 1], x


def test_level_ground_reads_position_and_stature():
    marks, hip, feet, box, x = scene(0.0); g = ground_path(marks, LINES[1] - LINES[0], hip, feet, box, np.ones(len(x), bool))
    assert abs(g["ground_rise_deg"]) < 0.05 and np.abs(g["sloped"]["X"] - x).max() < 0.005
    assert abs(g["sloped"]["camera_height_m"] - CAM_H) < 0.005 and abs(g["sloped"]["stature_m"] - STATURE) < 0.005
    assert np.allclose(g["sloped"]["lines_m"], LINES, atol=0.005)


def test_rising_ground_is_found_from_the_walkers_size():
    marks, hip, feet, box, x = scene(2.5, yaw_deg=4.0); g = ground_path(marks, LINES[1] - LINES[0], hip, feet, box, np.ones(len(x), bool))
    lvl = g["level"]["X"]; a = np.polyfit(x, lvl, 1); bow = np.sqrt(np.mean((lvl - np.polyval(a, x)) ** 2))
    X = g["sloped"]["X"]; b = np.polyfit(x, X, 1); res = np.sqrt(np.mean((X - np.polyval(b, x)) ** 2))
    assert abs(g["ground_rise_deg"] - 2.5) < 0.3
    assert abs(b[0] - 1) < 0.01 and res < 0.01 and bow > 3 * res                       # the level reading bows, this one does not
    assert g["sloped"]["stature_sd_m"] < 0.2 * g["level"]["stature_sd_m"] and abs(g["sloped"]["stature_m"] - STATURE) < 0.03


def test_marks_in_the_other_order_flip_the_axis():
    marks, hip, feet, box, x = scene(0.0); g = ground_path(marks[[2, 3, 0, 1]], LINES[1] - LINES[0], hip, feet, box, np.ones(len(x), bool))
    assert np.abs(g["sloped"]["X"] + x).max() < 0.005 and abs(g["sloped"]["camera_height_m"] - CAM_H) < 0.005


def test_score_reads_a_path_that_is_ten_percent_long():
    _, _, _, _, x = scene(0.0); rng = np.random.default_rng(0); turns = find_turns(x, FPS); ok = np.ones(len(x), bool)
    assert len(turns) == 6 and [s for _, s in turns] == [1, -1, 1, -1, 1, -1]
    u = np.array([np.cos(0.7), np.sin(0.7)]); n = np.array([-u[1], u[0]])
    P = 3.0 + np.outer(-1.1 * x, u) + np.outer(rng.normal(0, 0.05, len(x)), n)         # another axis, the other way round
    edge = np.zeros(len(x), bool); edge[turns[0][0]] = True; r = score(P, x, ok, turns, FPS, edge)
    assert abs(r["slope"] - 1.1) < 0.005 and r["rms_m"] < 0.01 and abs(r["off_axis_rms_m"] - 0.05) < 0.01
    assert all(abs(p["est_m"] / p["truth_m"] - 1.1) < 0.01 for p in r["passes"]) and len(r["passes"]) == 5
    assert [p["truth_is_lower_bound"] for p in r["passes"]] == [True, False, False, False, False]
    assert score(P[:50], x[:50], ok[:50], turns[:1], FPS, edge[:50]) is None


def test_a_known_height_corrects_a_wrong_distance():
    marks, hip, feet, box, x = scene(2.5); d = LINES[1] - LINES[0]; wrong = 0.92 * d
    g = ground_path(marks, wrong, hip, feet, box, np.ones(len(x), bool))["sloped"]; r = height_reading(g, wrong, STATURE / BOX_OVER_HEIGHT)
    assert abs(g["stature_m"] - 0.92 * STATURE) < 0.03                                   # the short tape shrinks the walker with it
    assert abs(r["distance_m"] - d) < 0.05 and abs(r["camera_height_m"] - CAM_H) < 0.01 and abs(r["slope_factor"] - 0.92) < 0.015
    assert abs(r["distance_in_boxes"] - d / STATURE) < 0.03
    same = height_reading(ground_path(marks, d, hip, feet, box, np.ones(len(x), bool))["sloped"], d, STATURE / BOX_OVER_HEIGHT)
    assert abs(same["slope_factor"] - 1) < 0.015 and abs(same["distance_m"] - r["distance_m"]) < 1e-6    # the tape drops out
