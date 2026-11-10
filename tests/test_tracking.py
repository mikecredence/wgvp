"""Colour descriptor on bodies cut by the top of the frame, and appearance cuts of a track."""
import numpy as np

from wgvp.tracking import GRID_NX, GRID_NY, TrackSet, split_fragments, torso_histogram

FPS = 30.0


def _kp(shoulder_y, shoulder_conf, hip_y, knee_y, head_conf=0.05, cx=150.0):
    kp = np.zeros((17, 3), np.float32)
    kp[:5] = [cx, 0.0, head_conf]
    kp[5], kp[6] = [cx - 15, shoulder_y, shoulder_conf], [cx + 15, shoulder_y, shoulder_conf]
    kp[11], kp[12] = [cx - 10, hip_y, 0.9], [cx + 10, hip_y, 0.9]
    kp[13], kp[14] = [cx - 10, knee_y, 0.9], [cx + 10, knee_y, 0.9]
    return kp


def test_descriptor_needs_a_torso_in_the_picture():
    img = np.zeros((200, 300, 3), np.uint8); img[:] = (40, 160, 40)
    box = np.array([120.0, 0.0, 180.0, 190.0])
    # torso in the picture, head above it: shoulders..hips
    assert torso_histogram(img, _kp(15, 0.8, 70, 120), box).sum() > 0.99
    # legs only: the pose model clamps the shoulders to the top edge with a passing confidence, the strip is thin
    assert torso_histogram(img, _kp(0, 0.5, 25, 80), box).sum() == 0
    # shoulders not seen, hips low enough for the column to hold the garment
    assert torso_histogram(img, _kp(0, 0.05, 60, 110), box).sum() > 0.99
    # the same column, too short against the thigh
    assert torso_histogram(img, _kp(0, 0.05, 20, 80), box).sum() == 0
    # a box that is not cut by the top keeps the earlier behaviour
    assert torso_histogram(img, _kp(40, 0.05, 60, 110, head_conf=0.9), np.array([120.0, 20.0, 180.0, 190.0])).sum() > 0.99


def _trackset(n, other=None):
    """One track of n detections whose descriptor changes half way; optionally a second id beside the change."""
    F = n + 10; frame = np.arange(n); tid = np.ones(n, int)
    box = np.tile(np.array([100.0, 50.0, 160.0, 250.0]), (n, 1))
    app = np.zeros((n, 48), np.float32); app[: n // 2, 3] = 1.0; app[n // 2:, 30] = 1.0
    if other is not None:
        frame = np.concatenate([frame, other]); tid = np.concatenate([tid, np.full(len(other), 2)])
        box = np.concatenate([box, np.tile(np.array([130.0, 40.0, 200.0, 260.0]), (len(other), 1))])
        a2 = np.zeros((len(other), 48), np.float32); a2[:, 30] = 1.0; app = np.concatenate([app, a2])
    N = len(frame)
    return TrackSet(video="v", view_id="full", box=[0, 0, 640, 480], width=640, height=480, fps_nominal=FPS, frame_idx=np.arange(F),
                    pts_ms=np.arange(F) / FPS * 1000.0, det_frame=frame, det_tid=tid, det_box=box, det_conf=np.full(N, 0.9),
                    det_kp=np.zeros((N, 17, 3), np.float32), det_app=app, det_vis=np.full(N, 2), grid=np.zeros((F, GRID_NY, GRID_NX), np.float32))


def test_appearance_cut_needs_a_second_person():
    lone = split_fragments(_trackset(120))
    assert len(lone.track_ids()) == 1                      # a lone person's descriptor changed on its own: no cut
    two = split_fragments(_trackset(120, other=np.arange(20, 50)))
    assert len(np.unique(two.det_tid[:120])) == 2          # a second id stood in that box 10 frames earlier: identity switch
    far = split_fragments(_trackset(400, other=np.arange(0, 30)))
    assert len(np.unique(far.det_tid[:400])) == 1          # the second id left more than 2 s before the change


def _frag(f0, f1, box, with_app):
    pos = np.arange(f0, f1); n = len(pos); app = np.zeros((n, 48), np.float32)
    if with_app:
        app[:, 3] = 1.0
    return {"tr": {"pos": pos, "pts_ms": pos / FPS * 1000.0, "box": np.tile(np.asarray(box, float), (n, 1)), "app": app}, "match": {"ok": False}}


def test_fragments_without_a_descriptor_attach_by_continuity():
    from wgvp.selection import DEFAULTS, _chain_blind
    body = [100, 0, 160, 180]
    frs = {1: _frag(0, 100, body, True),                    # the subject, torso in the picture
           2: _frag(100, 150, body, False),                 # the same box going on, legs only
           3: _frag(100, 150, [165, 0, 185, 40], False),    # a small static false detection beside it
           4: _frag(150, 200, body, False),                 # continues fragment 2
           5: _frag(400, 450, body, False)}                 # the same place, long after
    chained, masks = _chain_blind(frs, {1}, set(), 0.0, DEFAULTS)
    assert chained == {2, 4} and not masks
    # a second box on the subject fills only the frames the subject lacks
    frs[6] = _frag(90, 110, body, False)
    chained, masks = _chain_blind({k: frs[k] for k in (1, 6)}, {1}, set(), 0.0, DEFAULTS)
    assert chained == {6} and masks[6].sum() == 10
