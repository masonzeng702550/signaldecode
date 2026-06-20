"""Built-in Morse / CW decoder: envelope -> dot/dash -> dictionary.

Approach:
1. Compute the smoothed energy envelope and threshold it into keyed on/off.
2. Measure on-run and off-run lengths.
3. Cluster on-runs into dots vs dashes and off-runs into element/letter/word gaps.
4. Translate symbols to text via the international Morse table.
"""

from __future__ import annotations

import numpy as np

from .base import DecodeResult

MORSE_TABLE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", "-..-.": "/", "-....-": "-",
    "-.--.": "(", "-.--.-": ")", ".----.": "'", "---...": ":", "-.-.-.": ";",
    "-...-": "=", ".-.-.": "+", ".-..-.": '"', "...-..-": "$", ".--.-.": "@",
    "-.-.--": "!", "..--.-": "_",
}


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    frame = max(1, int(sr * 0.005))
    env = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(frame) / frame, "same"))
    if env.max() > 0:
        env = env / env.max()
    return env


def _runs(mask: np.ndarray):
    runs = []
    if len(mask) == 0:
        return runs
    cur = bool(mask[0])
    count = 1
    for v in mask[1:]:
        if bool(v) == cur:
            count += 1
        else:
            runs.append((cur, count))
            cur = bool(v)
            count = 1
    runs.append((cur, count))
    return runs


def _split_threshold(values: np.ndarray) -> float:
    """Pick a split point between two clusters (dot/dash) via a 1D 2-means pass."""
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return hi + 1.0
    c0, c1 = lo, hi
    for _ in range(20):
        a = values[np.abs(values - c0) <= np.abs(values - c1)]
        b = values[np.abs(values - c0) > np.abs(values - c1)]
        nc0 = a.mean() if len(a) else c0
        nc1 = b.mean() if len(b) else c1
        if abs(nc0 - c0) < 1e-6 and abs(nc1 - c1) < 1e-6:
            break
        c0, c1 = nc0, nc1
    return (c0 + c1) / 2.0


def decode(signal, threshold: float = 0.25) -> DecodeResult:
    x = signal.mono
    sr = signal.sr
    env = _envelope(x, sr)
    keyed = env > threshold
    runs = _runs(keyed)
    on_runs = np.array([n for v, n in runs if v], dtype=float)
    off_runs = np.array([n for v, n in runs if not v], dtype=float)

    if len(on_runs) < 2:
        return DecodeResult("morse", "builtin", 0.0, "", note="no keying detected")

    dot_dash_split = _split_threshold(on_runs)
    # Unit length ~ shortest reliable on-run (a dot).
    dot_len = np.median(on_runs[on_runs <= dot_dash_split])
    if not np.isfinite(dot_len) or dot_len <= 0:
        dot_len = on_runs.min()

    symbols = []
    confs = []
    for i, (v, n) in enumerate(runs):
        if v:
            symbols.append("-" if n > dot_dash_split else ".")
            # Confidence: how far this run sits from the split (clean separation).
            confs.append(min(1.0, abs(n - dot_dash_split) / max(dot_len, 1.0)))
        else:
            if i == 0 or i == len(runs) - 1:
                continue  # leading/trailing silence
            if n > dot_len * 5:
                symbols.append(" / ")  # word gap
            elif n > dot_len * 2:
                symbols.append(" ")    # letter gap
            # else: intra-character gap, no separator

    morse = "".join(symbols).strip()
    text = _morse_to_text(morse)
    confidence = float(np.clip(np.mean(confs) if confs else 0.0, 0.0, 1.0))
    # Penalise if too many letters failed to decode.
    if text:
        bad = text.count("�")
        confidence *= max(0.0, 1.0 - bad / max(1, len(text)))
    return DecodeResult("morse", "builtin", confidence, text, note=f"wpm~{_wpm(dot_len, sr):.0f}")


def _wpm(dot_len_samples: float, sr: int) -> float:
    dot_s = dot_len_samples / sr if sr else 0
    return 1.2 / dot_s if dot_s > 0 else 0.0


def _morse_to_text(morse: str) -> str:
    words = morse.split(" / ")
    out_words = []
    for w in words:
        letters = [c for c in w.split(" ") if c]
        out = "".join(MORSE_TABLE.get(l, "�") for l in letters)
        out_words.append(out)
    return " ".join(out_words).strip()
