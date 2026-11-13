"""The progress bar's number: it follows the pipeline's events, never falls, and ends at the end."""
from wgvp.progress import CPU_SHARE_WITH_LIFT, RunProgress


def _feed(rp, events):
    out = []
    for e in events:
        out.append(rp.update(*e)[0])
    return out


def test_one_session_moves_with_the_frames_not_in_one_jump():
    rp = RunProgress()
    ev = [("plan", 0, 1, ""), ("root", 0, 1, "OWN03"), ("video", 0, 1, "OWN03_SP.mp4")]
    ev += [("frames", k, 900, "") for k in range(0, 901, 9)] + [("session", 0, 1, "OWN03: OWN03_selfpace"), ("wrap", 0, 1, "")]
    f = _feed(rp, ev)
    assert all(b >= a for a, b in zip(f, f[1:]))
    assert max(b - a for a, b in zip(f, f[1:])) < 0.2          # no step takes a fifth of the bar
    half = RunProgress(); _feed(half, ev[:3] + [("frames", 450, 900, "")])
    assert 0.3 < half.fraction < 0.5                            # halfway through the frames is near the middle of the bar
    assert 0.9 < f[-1] < 1.0


def test_two_sessions_share_the_bar_and_text_names_the_work():
    rp = RunProgress()
    _feed(rp, [("plan", 0, 2, ""), ("root", 0, 2, "OWN01"), ("video", 0, 1, "a.mp4"), ("frames", 200, 400, "")])
    assert 0.15 < rp.fraction < 0.3 and "frame 200 of 400" in rp.text and "session 1 of 2: OWN01" in rp.text
    _feed(rp, [("session", 0, 1, "OWN01: OWN01_selfpace"), ("root", 1, 2, "OWN02")])
    assert 0.45 < rp.fraction < 0.52 and rp.text == "session 2 of 2: OWN02"


def test_a_cached_session_and_a_late_event_never_move_the_bar_back():
    rp = RunProgress()
    _feed(rp, [("plan", 0, 1, ""), ("root", 0, 1, "S"), ("video", 0, 1, "v.mp4"), ("session", 0, 2, "S: r1"), ("session", 1, 2, "S: r2")])
    high = rp.fraction
    assert rp.update("frames", 1, 100, "")[0] == high


def test_with_the_lift_the_cpu_stages_take_their_share_and_pieces_fill_the_rest():
    rp = RunProgress()
    _feed(rp, [("plan", 1, 1, ""), ("root", 0, 1, "S"), ("video", 0, 1, "v.mp4"), ("frames", 100, 100, ""), ("session", 0, 1, "S: r")])
    assert rp.fraction <= CPU_SHARE_WITH_LIFT
    a = rp.update("lift", 0, 4, "left/r/chunk_000")[0]; b = rp.update("lift", 2, 4, "")[0]; c = rp.update("lift", 4, 4, "")[0]
    assert a < b < c < 1.0 and "2 of 4" not in rp.text and "4 of 4" in rp.text


def test_sessions_take_a_share_of_the_bar_in_proportion_to_their_video_size():
    rp = RunProgress()
    _feed(rp, [("plan", 0, 2, "1000,3000"), ("root", 0, 2, "small"), ("video", 0, 1, "a.mp4"), ("session", 0, 1, "small: r"), ("root", 1, 2, "big")])
    assert 0.2 < rp.fraction < 0.27                                # a quarter of the videos, a quarter of the bar
    assert RunProgress().update("plan", 0, 2, "not,numbers")[0] == 0.0
