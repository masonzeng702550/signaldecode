"""Built-in DTMF decoder using the Goertzel algorithm.

Slides a ~40 ms window over the signal; in each frame it measures the energy at
the eight DTMF frequencies (four low-group, four high-group). A digit is emitted
when exactly one low and one high tone dominate. Consecutive identical frames are
collapsed and short gaps between presses are used as digit boundaries.
"""

from __future__ import annotations

import numpy as np

from .base import DecodeResult

LOW = (697, 770, 852, 941)
HIGH = (1209, 1336, 1477, 1633)
KEYPAD = {
    (697, 1209): "1", (697, 1336): "2", (697, 1477): "3", (697, 1633): "A",
    (770, 1209): "4", (770, 1336): "5", (770, 1477): "6", (770, 1633): "B",
    (852, 1209): "7", (852, 1336): "8", (852, 1477): "9", (852, 1633): "C",
    (941, 1209): "*", (941, 1336): "0", (941, 1477): "#", (941, 1633): "D",
}


def _goertzel_power(frame: np.ndarray, sr: int, freq: float) -> float:
    n = len(frame)
    k = int(0.5 + (n * freq) / sr)
    w = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(w)
    s_prev = s_prev2 = 0.0
    for sample in frame:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    return s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2


def decode(signal) -> DecodeResult:
    x = signal.mono
    sr = signal.sr
    win = int(sr * 0.04)
    if win < 8 or len(x) < win:
        return DecodeResult("dtmf", "builtin", 0.0, "", note="signal too short")
    hop = win // 2

    frames_digits = []
    frame_confs = []
    for start in range(0, len(x) - win, hop):
        frame = x[start : start + win]
        if np.sqrt(np.mean(frame ** 2)) < 1e-3:
            frames_digits.append(None)
            continue
        low_p = [_goertzel_power(frame, sr, f) for f in LOW]
        high_p = [_goertzel_power(frame, sr, f) for f in HIGH]
        li = int(np.argmax(low_p))
        hi = int(np.argmax(high_p))
        # Require the winners to clearly dominate their groups.
        low_ok = low_p[li] > 2.0 * (sum(low_p) - low_p[li]) / 3.0
        high_ok = high_p[hi] > 2.0 * (sum(high_p) - high_p[hi]) / 3.0
        if low_ok and high_ok:
            digit = KEYPAD[(LOW[li], HIGH[hi])]
            frames_digits.append(digit)
            total = sum(low_p) + sum(high_p)
            frame_confs.append((low_p[li] + high_p[hi]) / total if total else 0.0)
        else:
            frames_digits.append(None)

    digits = _collapse(frames_digits)
    text = "".join(digits)
    conf = float(np.clip(np.mean(frame_confs), 0.0, 1.0)) if frame_confs else 0.0
    note = f"{len(digits)} digits" if digits else "no DTMF tones detected"
    return DecodeResult("dtmf", "builtin", conf, text, note=note)


def _collapse(frames):
    """Collapse runs of identical digits separated by None gaps into presses."""
    out = []
    prev = None
    for d in frames:
        if d is None:
            prev = None
            continue
        if d != prev:
            out.append(d)
        prev = d
    return out
