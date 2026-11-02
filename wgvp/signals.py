"""Signal processing shared by selection, sync and reconciliation (pure numpy/scipy, CPU).

Conventions
-----------
* All matching is done on a uniform grid at FS_MATCH Hz (default 100 Hz, the IMU rate).
* ``offset`` (seconds) is the VIDEO time at which IMU time 0 occurs:  t_video = t_imu + offset.
  A positive offset means the video started before the IMU recording.
* Video-side signals are built from pixel keypoints on the video's PTS time base (so variable frame rate and
  dropped frames are handled by interpolation onto the grid, with gaps larger than ``max_gap_s`` left NaN).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS_MATCH = 100.0
GAIT_BAND = (0.5, 5.0)      # Hz: stride (~1 Hz) and step (~2 Hz) harmonics
ENV_WIN_S = 1.0             # RMS envelope window
ENV_FS = 20.0               # envelope grid for the coarse search


def bandpass(x: np.ndarray, fs: float, lo: float = GAIT_BAND[0], hi: float = GAIT_BAND[1], order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass. NaNs are linearly bridged before filtering and restored after."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.full_like(x, np.nan)
    xi = x.copy()
    if not ok.all():
        idx = np.arange(len(x))
        xi[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    y = sosfiltfilt(sos, xi - np.nanmean(xi))
    y[~ok] = np.nan
    return y


def highpass(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    sos = butter(order, fc / (fs / 2), btype="high", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def double_integrate(acc: np.ndarray, fs: float, lo: float = GAIT_BAND[0], hi: float = GAIT_BAND[1]) -> np.ndarray:
    """Acceleration -> displacement inside the gait band (high-pass after each integration keeps drift out).

    Within a pass band each integration multiplies the spectrum by 1/(i*omega), so the result is in phase with the
    true displacement (two integrations: -1/omega^2, i.e. displacement = -acc/omega^2 for a sinusoid, which is what
    integrating twice gives). Units are arbitrary; matching is scale-free.
    """
    a = bandpass(acc, fs, lo, hi)
    ok = np.isfinite(a)
    a = np.where(ok, a, 0.0)
    v = highpass(np.cumsum(a) / fs, fs, lo)
    d = highpass(np.cumsum(v) / fs, fs, lo)
    d = bandpass(d, fs, lo, hi)
    d[~ok] = np.nan
    return d


def rms_envelope(x: np.ndarray, fs: float, win_s: float = ENV_WIN_S) -> np.ndarray:
    """Moving RMS (NaN-aware, NaN where the window has < 50% valid samples)."""
    x = np.asarray(x, dtype=np.float64)
    n = max(3, int(round(win_s * fs)))
    ok = np.isfinite(x).astype(np.float64)
    x2 = np.where(ok > 0, x, 0.0) ** 2
    k = np.ones(n) / n
    num = np.convolve(x2, k, mode="same")
    den = np.convolve(ok, k, mode="same")
    env = np.sqrt(num / np.maximum(den, 1e-9))
    env[den < 0.5] = np.nan
    return env


def resample_to_grid(t: np.ndarray, x: np.ndarray, fs: float, t0: float = 0.0, t1: float | None = None,
                     max_gap_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of (t, x) onto a uniform grid; grid points farther than max_gap_s from any sample, or
    inside a sample gap longer than max_gap_s, become NaN. Returns (grid_t, grid_x)."""
    t = np.asarray(t, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if t1 is None:
        t1 = float(t.max()) if len(t) else t0
    grid = np.arange(t0, t1 + 1e-9, 1.0 / fs)
    if len(t) < 2:
        return grid, np.full(len(grid), np.nan)
    order = np.argsort(t); t, x = t[order], x[order]
    y = np.interp(grid, t, x, left=np.nan, right=np.nan)
    # mask gaps
    gaps = np.diff(t)
    big = np.where(gaps > max_gap_s)[0]
    for i in big:
        y[(grid > t[i]) & (grid < t[i + 1])] = np.nan
    return grid, y


def masked_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int, min_overlap: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-lag Pearson correlation of two NaN-masked signals at integer lags k in [-max_lag, max_lag].

    r[k] = corr(a[t], b[t - k]) over the t where both are valid, i.e. a is b delayed by k samples (a = video,
    b = IMU, k = offset in samples). Means and variances are taken inside each overlap (true Pearson), so partial
    overlaps at the edges are not inflated. FFT-based: six products, exact. Returns (lags, r, n_overlap); r is NaN
    where n_overlap < min_overlap.
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ma = np.isfinite(a); mb = np.isfinite(b)
    a0 = np.where(ma, a, 0.0); b0 = np.where(mb, b, 0.0)
    n = len(a0) + len(b0)
    nfft = 1 << int(np.ceil(np.log2(max(n + 1, 2))))
    FA, FB = np.fft.rfft(a0, nfft), np.fft.rfft(b0, nfft)
    FA2, FB2 = np.fft.rfft(a0 ** 2, nfft), np.fft.rfft(b0 ** 2, nfft)
    FMA, FMB = np.fft.rfft(ma.astype(np.float64), nfft), np.fft.rfft(mb.astype(np.float64), nfft)
    cj = np.conj
    sab = np.fft.irfft(FA * cj(FB), nfft)      # sum a[t] b[t-k]
    sa = np.fft.irfft(FA * cj(FMB), nfft)      # sum a[t] mb[t-k]
    sb = np.fft.irfft(FMA * cj(FB), nfft)      # sum ma[t] b[t-k]
    saa = np.fft.irfft(FA2 * cj(FMB), nfft)    # sum a[t]^2 mb[t-k]
    sbb = np.fft.irfft(FMA * cj(FB2), nfft)    # sum ma[t] b[t-k]^2
    nn = np.fft.irfft(FMA * cj(FMB), nfft)     # overlap count
    lags = np.arange(-max_lag, max_lag + 1)
    idx = lags % nfft
    linear = (lags >= -(len(b0) - 1)) & (lags <= len(a0) - 1)     # outside: no overlap (the circular FFT would alias)
    N = np.maximum(np.round(nn[idx]), 1.0)
    cov = sab[idx] - sa[idx] * sb[idx] / N
    va = saa[idx] - sa[idx] ** 2 / N
    vb = sbb[idx] - sb[idx] ** 2 / N
    den = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    cnt = np.where(linear, np.round(nn[idx]).astype(int), 0)
    r = np.where(linear & (den > 1e-12) & (cnt >= min_overlap), cov / np.maximum(den, 1e-12), np.nan)
    return lags, r, cnt


def parabolic_peak(r: np.ndarray, i: int) -> float:
    """Sub-sample peak location by parabolic interpolation around index i (returns fractional index)."""
    if i <= 0 or i >= len(r) - 1 or not np.isfinite(r[i - 1]) or not np.isfinite(r[i + 1]):
        return float(i)
    y0, y1, y2 = r[i - 1], r[i], r[i + 1]
    d = y0 - 2 * y1 + y2
    if abs(d) < 1e-12:
        return float(i)
    return float(i + 0.5 * (y0 - y2) / d)


def match_signals(video_t: np.ndarray, video_x: np.ndarray, imu_x: np.ndarray, fs: float = FS_MATCH,
                  env_fs: float = ENV_FS, fine_window_s: float = 0.75, min_overlap_s: float = 2.0,
                  lo: float = GAIT_BAND[0], hi: float = GAIT_BAND[1]) -> dict:
    """Two-scale match of a video-side oscillation (on PTS time) against an IMU-side reference on a uniform grid.

    1. coarse: RMS envelopes (activity pattern: stand / walk / turn), unambiguous but broad;
    2. fine: band-passed oscillation inside +-fine_window_s of the coarse lag (sub-sample via parabola).
    Returns offset_s (video time of IMU t=0), r_fine, r_env, the global fine peak outside the coarse window
    (ambiguity ratio), overlap seconds, and the full fine correlation function for joint estimates.
    """
    out = {"ok": False, "offset_s": np.nan, "r_fine": np.nan, "r_env": np.nan, "r_fine_outside": np.nan,
           "ambiguity_ratio": np.nan, "overlap_s": 0.0, "coarse_offset_s": np.nan, "reason": ""}
    imu_x = np.asarray(imu_x, dtype=np.float64)
    n_imu = len(imu_x)
    t1 = float(np.nanmax(video_t)) if len(video_t) else 0.0
    grid, v = resample_to_grid(video_t, video_x, fs, 0.0, t1)
    if np.isfinite(v).sum() < min_overlap_s * fs:
        out["reason"] = "video signal too short"; return out
    vb = bandpass(v, fs, lo, hi)
    ub = bandpass(imu_x, fs, lo, hi) if not _looks_bandpassed(imu_x) else np.asarray(imu_x, dtype=np.float64)
    # --- coarse on envelopes
    dec = int(round(fs / env_fs))
    ev = rms_envelope(vb, fs)[::dec]; eu = rms_envelope(ub, fs)[::dec]
    ev = ev - np.nanmean(ev); eu = eu - np.nanmean(eu)
    max_lag_env = len(ev) + len(eu)
    n_valid_u = int(np.isfinite(eu).sum()); n_valid_v = int(np.isfinite(ev).sum())
    n_ref = min(n_valid_u, n_valid_v)
    lags_e, r_e, n_e = masked_xcorr(ev, eu, max_lag_env, min_overlap=max(int(min_overlap_s * env_fs), int(0.25 * n_ref)))
    if not np.isfinite(r_e).any():
        out["reason"] = "no envelope overlap"; return out
    # prefer lags where the shorter signal is (nearly) fully inside the longer one
    score_e = r_e * np.sqrt(np.minimum(1.0, n_e / max(n_ref, 1)))
    ie = int(np.nanargmax(score_e)); coarse = lags_e[ie] / env_fs
    out["coarse_offset_s"] = float(coarse); out["r_env_coarse"] = float(r_e[ie]); out["env_overlap_frac"] = float(n_e[ie] / max(n_ref, 1))
    # --- fine on the oscillation over the full lag range, weighted by the envelope agreement at each lag: when the
    # activity pattern is informative (stand / walk / turn) it breaks the stride-period aliases of the periodic fine
    # correlation; when it is flat (continuous walking) the weight is ~constant and the fine peak decides alone.
    max_lag = len(vb) + n_imu
    n_ref_f = min(int(np.isfinite(vb).sum()), int(np.isfinite(ub).sum()))
    lags_f, r_f, n_f = masked_xcorr(vb, ub, max_lag, min_overlap=max(int(min_overlap_s * fs), int(0.25 * n_ref_f)))
    if not np.isfinite(r_f).any():
        out["reason"] = "no oscillation overlap"; return out
    r_e_fine = np.interp(lags_f / fs, lags_e / env_fs, np.nan_to_num(r_e, nan=0.0), left=0.0, right=0.0)
    w = 0.5 + 0.5 * np.clip(r_e_fine, 0.0, 1.0)
    score = np.where(np.isfinite(r_f), r_f * w, np.nan)
    i_best = int(np.nanargmax(score))
    frac = parabolic_peak(r_f, i_best)
    off = (lags_f[0] + frac) / fs
    far = np.abs(lags_f - lags_f[i_best]) > 0.35 * fs
    s_out = np.where(far, score, np.nan)
    score_outside = float(np.nanmax(s_out)) if np.isfinite(s_out).any() else np.nan
    out.update(ok=True, offset_s=float(off), r_fine=float(r_f[i_best]), r_env=float(r_e_fine[i_best]), r_fine_outside=score_outside,
               ambiguity_ratio=float(score[i_best] / score_outside) if np.isfinite(score_outside) and score_outside > 0 else np.inf,
               overlap_s=float(n_f[i_best] / fs), lags_s=lags_f / fs, r_fine_fn=r_f, r_env_fn=r_e, lags_env_s=lags_e / env_fs)
    return out


def _looks_bandpassed(x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=np.float64)
    return bool(np.isfinite(x).any() and abs(np.nanmean(x)) < 1e-3 * (np.nanstd(x) + 1e-12))


def combined_lag_curve(curves: list[tuple[np.ndarray, np.ndarray, float]], fs: float = FS_MATCH) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sum of several (lags_s, r_fn, weight) correlation functions on a common lag axis."""
    lo = min(c[0][0] for c in curves); hi = max(c[0][-1] for c in curves)
    axis = np.arange(round(lo * fs), round(hi * fs) + 1) / fs
    acc = np.zeros(len(axis)); wsum = np.zeros(len(axis))
    for lags, r, w in curves:
        i0 = int(round((lags[0] - axis[0]) * fs))
        rr = np.nan_to_num(r, nan=0.0)
        acc[i0:i0 + len(rr)] += w * rr; wsum[i0:i0 + len(rr)] += w * np.isfinite(r)
    return axis, np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)
