import math
from typing import Optional

import numpy as np
import soundfile as sf
from scipy import signal


def _prep_env(x: np.ndarray, sr: int, *, max_seconds: float, target_sr: float = 400.0) -> tuple[np.ndarray, float]:
    """Return (env, env_sr) onset-like envelope, downsampled for speed."""
    # trim
    nmax = int(sr * max_seconds)
    if x.size > nmax:
        x = x[:nmax]

    # High-pass a bit to reduce DC/rumble
    try:
        b, a = signal.butter(2, 40 / (sr / 2), btype="highpass")
        x = signal.lfilter(b, a, x)
    except Exception:
        pass

    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))

    # Onset envelope: energy of first difference of rectified signal
    env = np.abs(np.diff(x))

    # Smooth envelope ~ 10ms
    win = max(1, int(sr * 0.01))
    if win > 1:
        env = np.convolve(env, np.ones(win, dtype=np.float32) / win, mode="same")

    # Downsample envelope for speed/precision tradeoff
    decim = max(1, int(sr / target_sr))
    env = env[::decim]
    env_sr = sr / decim

    # normalize (z-score)
    env = env - float(np.mean(env))
    env = env / (float(np.std(env)) + 1e-8)
    return env.astype(np.float32, copy=False), float(env_sr)


def _find_rhythm_start(env: np.ndarray, env_sr: float) -> float:
    """Best-effort start time (seconds) when rhythm becomes "present/stable".

    Heuristic: find first region where the envelope energy stays above a threshold
    for a short sustained window.
    """
    try:
        # energy curve (moving RMS)
        win = max(1, int(env_sr * 0.25))  # 250ms
        if win > 1:
            e = np.sqrt(np.convolve(env * env, np.ones(win, dtype=np.float32) / win, mode="same"))
        else:
            e = np.abs(env)

        # baseline from early part
        head = e[: max(1, int(env_sr * 8))]  # first 8s
        base = float(np.median(head))
        spread = float(np.median(np.abs(head - base)) + 1e-6)

        # threshold: baseline + k * MAD
        thr = base + 4.0 * spread

        above = e > thr
        # require sustained activity
        sustain = max(1, int(env_sr * 1.0))  # 1s sustained
        if above.size < sustain + 5:
            return 0.0

        # Find first index where we have sustain consecutive True values
        run = 0
        for i, ok in enumerate(above):
            run = run + 1 if ok else 0
            if run >= sustain:
                # back up a bit so we start slightly before the sustained window
                start_i = max(0, i - sustain - int(env_sr * 0.25))
                return float(start_i / env_sr)

        return 0.0
    except Exception:
        return 0.0


def detect_bpm_and_offset_from_wav(
    path: str,
    *,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
    max_seconds: float = 120.0,
) -> tuple[Optional[float], float]:
    """Lightweight BPM detection with best-effort "rhythm start" offset.

    Returns (bpm_or_none, start_offset_seconds).
    """
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None, 0.0

    if x is None:
        return None, 0.0

    # mono
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = np.mean(x, axis=1)

    if not isinstance(x, np.ndarray) or x.size < sr * 5:
        return None, 0.0

    env, env_sr = _prep_env(x, sr, max_seconds=max_seconds, target_sr=400.0)
    if env.size < int(env_sr * 5):
        return None, 0.0

    start_s = _find_rhythm_start(env, env_sr)
    start_idx = int(max(0.0, start_s) * env_sr)

    # Use a window after the rhythm start for tempo detection (longer = more stable)
    seg = env[start_idx:]

    # Prefer ~60s after start when available
    want = int(env_sr * 60.0)
    if seg.size > want:
        seg = seg[:want]

    if seg.size < int(env_sr * 8):
        seg = env  # fallback
        start_s = 0.0
        start_idx = 0

    # Autocorrelation on segment
    ac = signal.fftconvolve(seg, seg[::-1], mode="full")
    ac = ac[ac.size // 2 :]

    # Limit lag range to bpm window
    min_lag = int(env_sr * 60.0 / max_bpm)
    max_lag = int(env_sr * 60.0 / min_bpm)
    if max_lag <= min_lag + 2 or max_lag >= ac.size:
        return None, float(start_s)

    window = ac[min_lag:max_lag]
    if window.size < 10:
        return None, float(start_s)

    # Pick top-K peaks to reduce octave errors
    k = min(8, window.size)
    top_idx = np.argpartition(window, -k)[-k:]
    top_idx = top_idx[np.argsort(window[top_idx])[::-1]]

    def _parabolic_peak(y: np.ndarray, i: int) -> float:
        """Refine peak index with a parabola fit around i (sub-sample)."""
        if i <= 0 or i >= y.size - 1:
            return float(i)
        y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) < 1e-12:
            return float(i)
        delta = 0.5 * (y0 - y2) / denom
        return float(i + delta)

    candidates_bpm: list[float] = []
    for idx in top_idx:
        lag_ref = _parabolic_peak(window, int(idx)) + min_lag
        if lag_ref <= 0:
            continue
        bpm0 = 60.0 * env_sr / float(lag_ref)
        candidates_bpm.extend([bpm0, bpm0 * 2.0, bpm0 / 2.0])

    center = (min_bpm + max_bpm) / 2.0
    valid = [c for c in candidates_bpm if min_bpm <= c <= max_bpm and math.isfinite(c)]
    if not valid:
        return None, float(start_s)

    # Score: prefer tempos that multiple peaks agree on, then closeness to center.
    def _score(b: float) -> float:
        strength = sum(1 for v in valid if abs(v - b) < 0.6)
        return -strength * 10.0 + abs(b - center)

    bpm2 = min(valid, key=_score)

    return round(float(bpm2), 2), float(round(start_s, 3))


def detect_bpm_from_wav(path: str, *, min_bpm: float = 60.0, max_bpm: float = 200.0, max_seconds: float = 120.0) -> Optional[float]:
    """Back-compat wrapper."""
    bpm, _off = detect_bpm_and_offset_from_wav(path, min_bpm=min_bpm, max_bpm=max_bpm, max_seconds=max_seconds)
    return bpm
