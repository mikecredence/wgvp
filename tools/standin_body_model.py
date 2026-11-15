"""A stand-in SMPL-X model file for tests and continuous integration.

The real SMPL-X model (smpl-x.is.tue.mpg.de) is licensed and cannot be redistributed, so the public tests cannot
carry it. This script writes `smplx/SMPLX_NEUTRAL.npz` with every array that the `smplx` package, GVHMR's loaders
and this pipeline read, under the real file's names and in its shapes (10,475 vertices, 55 joints, 400 shape and
486 pose directions), but with synthetic content: a body of capsules around the mean SMPL-X skeleton (the 22 body
joints at their rest positions, the hand and face joints placed from them), skin weights by limb, no pose blend
shapes, and ten small shape directions. Posing it gives the same joint angles and translations as posing the real
model; only the skin differs. Where a lifter corrects a pose through the model's joints (GVHMR anchors the feet on
its skeleton), the correction is made on this skeleton instead of the real one. It checks that the code runs on a
machine, not what the real model would render.

    python tools/standin_body_model.py <folder>        # writes <folder>/smplx/SMPLX_NEUTRAL.npz and a README
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V_SMPLX, J_SMPLX, N_SHAPE, N_POSE = 10475, 55, 400, 9 * 54
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
           20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]
# the 22 body joints of the mean neutral SMPL-X body, pelvis at the origin, y up, x to the subject's left
REST = np.array([[0.0, 0.0, 0.0], [0.0582, -0.0928, -0.026], [-0.0633, -0.1039, -0.0213], [-0.0028, 0.1099, -0.0276],
                 [0.1129, -0.4715, -0.0354], [-0.1075, -0.4663, -0.0381], [0.0067, 0.2417, -0.0336], [0.0694, -0.8746, -0.0673],
                 [-0.0921, -0.877, -0.0583], [-0.0046, 0.294, -0.0051], [0.1167, -0.9326, 0.0509], [-0.1309, -0.9353, 0.0608],
                 [-0.0168, 0.4591, -0.0367], [0.0417, 0.3789, -0.0123], [-0.0523, 0.3783, -0.0185], [0.008, 0.6196, -0.016],
                 [0.161, 0.4367, -0.0278], [-0.1549, 0.4318, -0.0312], [0.4151, 0.3645, -0.0703], [-0.4261, 0.3953, -0.0576],
                 [0.6671, 0.3877, -0.0727], [-0.6753, 0.3908, -0.073]])
AROUND, ALONG = 12, 6                                                  # vertices around a capsule, rings along it
# capsules: (from joint, to joint or point, driving joint, radius)
HEAD_TOP, HAND_L, HAND_R = "head_top", "hand_l", "hand_r"
POINTS = {HEAD_TOP: REST[15] + [0.0, 0.17, 0.0], HAND_L: REST[20] + [0.09, -0.01, 0.0], HAND_R: REST[21] + [-0.09, -0.01, 0.0]}
CAPSULES = [(0, 3, 0, 0.11), (3, 6, 3, 0.12), (6, 9, 6, 0.12), (9, 12, 9, 0.07), (12, 15, 12, 0.05), (15, HEAD_TOP, 15, 0.10),
            (0, 1, 0, 0.08), (0, 2, 0, 0.08), (1, 4, 1, 0.075), (2, 5, 2, 0.075), (4, 7, 4, 0.055), (5, 8, 5, 0.055),
            (7, 10, 7, 0.04), (8, 11, 8, 0.04), (9, 13, 9, 0.06), (9, 14, 9, 0.06), (13, 16, 13, 0.06), (14, 17, 14, 0.06),
            (16, 18, 16, 0.045), (17, 19, 17, 0.045), (18, 20, 18, 0.035), (19, 21, 19, 0.035), (20, HAND_L, 20, 0.03), (21, HAND_R, 21, 0.03)]
FINGERS = [("index", 40.0), ("middle", 15.0), ("ring", -15.0), ("pinky", -40.0), ("thumb", 75.0)]   # angle around the hand axis
FINGER_ORDER = ["index", "middle", "pinky", "ring", "thumb"]                                        # SMPL-X's order of hand joints


def _point(key) -> np.ndarray:
    return REST[key] if isinstance(key, int) else POINTS[key]


def build() -> dict:
    verts, faces, weights, caps, rings = [], [], [], {}, {}
    for a, b, driver, r in CAPSULES:
        pa, pb = _point(a), _point(b); u = pb - pa; u /= np.linalg.norm(u)
        z = np.array([0.0, 0.0, 1.0]); v = z - (z @ u) * u                 # theta = 0 points to the subject's front where it can
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0, 0.0, 0.0]) - u[0] * u
        v /= np.linalg.norm(v); w = np.cross(u, v)
        distal = b if isinstance(b, int) else driver
        base = len(verts); caps[(a, "a")] = base; verts.append(pa); weights.append((driver, 1.0))
        for k in range(ALONG):
            t = k / (ALONG - 1); c = pa + t * (pb - pa); s = 0.5 * max(0.0, (t - 0.75) / 0.25)   # a little blending to the next joint
            rings[(a, b, k)] = len(verts)
            for i in range(AROUND):
                th = 2 * np.pi * i / AROUND; verts.append(c + r * (np.cos(th) * v + np.sin(th) * w)); weights.append((driver, 1.0 - s, distal, s))
        caps[(b, "b")] = len(verts); verts.append(pb); weights.append((distal, 1.0))
        for k in range(ALONG - 1):
            r0, r1 = rings[(a, b, k)], rings[(a, b, k + 1)]
            for i in range(AROUND):
                j = (i + 1) % AROUND; faces += [[r0 + i, r0 + j, r1 + i], [r0 + j, r1 + j, r1 + i]]
        r0, r1 = rings[(a, b, 0)], rings[(a, b, ALONG - 1)]
        for i in range(AROUND):
            j = (i + 1) % AROUND; faces += [[base, r0 + j, r0 + i], [caps[(b, "b")], r1 + i, r1 + j]]
    n_mesh = len(verts)
    verts += [POINTS[HEAD_TOP]] * (V_SMPLX - n_mesh); weights += [(15, 1.0)] * (V_SMPLX - n_mesh)   # filler: the extra-joint vertex ids land on the head
    v_template = np.asarray(verts, np.float32); f = np.asarray(faces, np.uint32)
    W = np.zeros((V_SMPLX, J_SMPLX), np.float32)
    for i, wt in enumerate(weights):
        for j, s in zip(wt[::2], wt[1::2]):
            W[i, j] += s
    Jr = np.zeros((J_SMPLX, V_SMPLX), np.float32)
    for j in range(22):                                                    # each body joint: the cap vertex at it
        Jr[j, caps[(j, "a")] if (j, "a") in caps else caps[(j, "b")]] = 1.0
    ring_v = lambda a, b, k, deg: rings[(a, b, k)] + int(round(deg / 360.0 * AROUND)) % AROUND
    Jr[22, caps[(15, "a")]] = 0.5; Jr[22, ring_v(15, HEAD_TOP, 1, 0.0)] = 0.5                       # jaw: forward and a little up
    Jr[23, caps[(15, "a")]] = 0.4; Jr[23, ring_v(15, HEAD_TOP, 3, 30.0)] = 0.6                      # eyes
    Jr[24, caps[(15, "a")]] = 0.4; Jr[24, ring_v(15, HEAD_TOP, 3, -30.0)] = 0.6
    for side, wrist, hand in ((0, 20, HAND_L), (1, 21, HAND_R)):
        for fi, name in enumerate(FINGER_ORDER):
            deg = dict(FINGERS)[name]
            for i in range(3):
                k = 1.0 + 0.25 * i; j = 25 + 15 * side + 3 * fi + i             # beyond the hand's end, along the wrist -> ring-vertex line
                Jr[j, caps[(wrist, "a")]] = 1.0 - k; Jr[j, ring_v(wrist, hand, ALONG - 1, deg)] = k
    joints = np.einsum("jv,vc->jc", Jr.astype(np.float64), v_template.astype(np.float64))     # einsum: macOS Accelerate warns on matmul
    sd = np.zeros((V_SMPLX, 3, N_SHAPE), np.float64)
    pats = [np.eye(3), np.diag([0, 1, 0]), np.diag([1, 0, 1]), np.diag([1, 0, 0]), np.diag([0, 0, 1]),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]), np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])]
    for k, P in enumerate(pats):
        sd[:, :, k] = 0.01 * np.einsum("vc,dc->vd", v_template.astype(np.float64), P.astype(np.float64))
    sd[:, 1, 9] = 0.02 * v_template[:, 1].astype(np.float64) ** 2                                   # one that is not linear in the template
    kin = np.stack([np.asarray(PARENTS, np.int64), np.arange(J_SMPLX, dtype=np.int64)])
    n_f = len(f); lmk = (np.arange(51) * 7) % n_f; dyn = (np.arange(79 * 17).reshape(79, 17) * 11) % n_f
    bary = np.zeros((51, 3)); bary[:, 0] = 1.0; dbary = np.zeros((79, 17, 3)); dbary[:, :, 0] = 1.0
    return dict(v_template=v_template, f=f, ft=f.copy(), vt=np.zeros((V_SMPLX, 2), np.float64), weights=W, J_regressor=Jr, kintree_table=kin,
                shapedirs=sd, posedirs=np.zeros((V_SMPLX, 3, N_POSE), np.float32),
                hands_meanl=np.zeros(45, np.float32), hands_meanr=np.zeros(45, np.float32),
                hands_componentsl=np.eye(45, dtype=np.float32), hands_componentsr=np.eye(45, dtype=np.float32),
                hands_coeffsl=np.zeros((1554, 45), np.float32), hands_coeffsr=np.zeros((1554, 45), np.float32),
                lmk_faces_idx=lmk.astype(np.int64), lmk_bary_coords=bary, dynamic_lmk_faces_idx=dyn.astype(np.int64), dynamic_lmk_bary_coords=dbary,
                _joints=joints, _n_mesh=n_mesh)


def write(folder: Path) -> Path:
    d = build(); out = Path(folder) / "smplx"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "SMPLX_NEUTRAL.npz", **{k: v for k, v in d.items() if not k.startswith("_")})
    (Path(folder) / "README.txt").write_text(
        "STAND-IN body model, not SMPL-X. Written by tools/standin_body_model.py of the wgvp package for its tests: a capsule\n"
        "body on the mean SMPL-X skeleton in the real file's array names and shapes. It is not the licensed model and renders a\n"
        "capsule figure. Get the real SMPL-X model from smpl-x.is.tue.mpg.de.\n", encoding="utf-8")
    return out / "SMPLX_NEUTRAL.npz"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    p = write(Path(sys.argv[1])); d = build()
    print(f"{p}: {V_SMPLX} vertices ({d['_n_mesh']} in the mesh), {len(d['f'])} faces, {J_SMPLX} joints")
