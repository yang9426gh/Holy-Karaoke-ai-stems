from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

# Krumhansl-Schmuckler key profiles (major/minor)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_midi(f: np.ndarray) -> np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(f, 1e-9) / 440.0)


def detect_key_from_wav(
    path: str,
    *,
    max_seconds: float = 90.0,
    sr_target: int = 22050,
    fmin: float = 55.0,
    fmax: float = 1760.0,
) -> Optional[Tuple[str, str, float]]:
    """Return (tonic_note, mode, confidence) using a simple chroma+K-S correlation.

    mode is "major" or "minor".
    confidence is a 0..1-ish margin score.
    """
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None

    if x is None:
        return None

    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = np.mean(x, axis=1)

    if not isinstance(x, np.ndarray) or x.size < sr * 5:
        return None

    # trim
    nmax = int(sr * max_seconds)
    if x.size > nmax:
        x = x[:nmax]

    # resample (cheap polyphase)
    if sr != sr_target:
        g = math.gcd(sr, sr_target)
        up = sr_target // g
        down = sr // g
        x = signal.resample_poly(x, up, down).astype(np.float32)
        sr = sr_target

    # high-pass to reduce rumble and DC
    try:
        b, a = signal.butter(2, 40 / (sr / 2), btype="highpass")
        x = signal.lfilter(b, a, x)
    except Exception:
        pass

    x = x - float(np.mean(x))

    # STFT
    n_fft = 4096
    hop = 1024
    win = signal.windows.hann(n_fft, sym=False).astype(np.float32)

    # pad
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))

    # frames
    frames = []
    for start in range(0, x.size - n_fft + 1, hop):
        frames.append(x[start : start + n_fft] * win)
    if not frames:
        return None

    X = np.fft.rfft(np.stack(frames, axis=0), axis=1)
    mag = np.abs(X).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None

    freqs = freqs[mask]
    mag = mag[:, mask]

    midi = _hz_to_midi(freqs)
    pcs = np.mod(np.round(midi).astype(np.int32), 12)

    chroma = np.zeros(12, dtype=np.float32)
    # accumulate energy per pitch class
    for pc in range(12):
        idx = pcs == pc
        if np.any(idx):
            chroma[pc] = float(np.sum(mag[:, idx]))

    if float(np.sum(chroma)) <= 1e-6:
        return None

    # normalize
    chroma = chroma / (float(np.linalg.norm(chroma)) + 1e-9)

    # correlation over rotations
    maj = _MAJOR_PROFILE / (float(np.linalg.norm(_MAJOR_PROFILE)) + 1e-9)
    minp = _MINOR_PROFILE / (float(np.linalg.norm(_MINOR_PROFILE)) + 1e-9)

    scores = []
    for k in range(12):
        # rotate profiles so index 0 corresponds to tonic
        maj_k = np.roll(maj, k)
        min_k = np.roll(minp, k)
        smaj = float(np.dot(chroma, maj_k))
        smin = float(np.dot(chroma, min_k))
        scores.append((smaj, smin))

    # best
    best_k = 0
    best_mode = "major"
    best_score = -1e9
    all_flat = []
    for k, (smaj, smin) in enumerate(scores):
        all_flat.append((smaj, k, "major"))
        all_flat.append((smin, k, "minor"))
        if smaj > best_score:
            best_score = smaj
            best_k = k
            best_mode = "major"
        if smin > best_score:
            best_score = smin
            best_k = k
            best_mode = "minor"

    all_flat.sort(reverse=True, key=lambda x: x[0])
    runner = all_flat[1][0] if len(all_flat) > 1 else -1e9
    margin = max(0.0, best_score - runner)
    conf = float(min(1.0, margin / 0.15))  # heuristic

    return NOTES[best_k], best_mode, conf
