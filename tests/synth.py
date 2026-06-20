"""Synthetic signal generators for deterministic, offline tests.

Produces known Morse, DTMF and FSK waveforms so decoders/features can be
validated without depending on an external CTF corpus.
"""

from __future__ import annotations

import numpy as np

SR = 44100

MORSE_TABLE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}

DTMF_FREQS = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477),
}


def _tone(freq: float, dur_s: float, sr: int = SR, amp: float = 0.7) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(dur_s * sr), dtype=float)


def morse_wave(text: str, freq: float = 700.0, wpm: float = 18.0, sr: int = SR) -> np.ndarray:
    """Generate a Morse/CW waveform for ``text`` (A-Z, 0-9, spaces)."""
    dot = 1.2 / wpm  # PARIS timing
    parts = []
    for word in text.upper().split(" "):
        for ch in word:
            code = MORSE_TABLE.get(ch)
            if code is None:
                continue
            for sym in code:
                parts.append(_tone(freq, dot if sym == "." else dot * 3, sr))
                parts.append(_silence(dot, sr))  # intra-char gap
            parts.append(_silence(dot * 2, sr))  # letter gap (total ~3)
        parts.append(_silence(dot * 4, sr))      # word gap (total ~7)
    return np.concatenate([_silence(0.05, sr)] + parts + [_silence(0.05, sr)])


def dtmf_wave(digits: str, tone_s: float = 0.12, gap_s: float = 0.08,
              sr: int = SR) -> np.ndarray:
    """Generate a DTMF dial-tone sequence for ``digits``."""
    parts = [_silence(0.05, sr)]
    for d in digits:
        if d not in DTMF_FREQS:
            continue
        lo, hi = DTMF_FREQS[d]
        parts.append((_tone(lo, tone_s, sr, 0.5) + _tone(hi, tone_s, sr, 0.5)))
        parts.append(_silence(gap_s, sr))
    return np.concatenate(parts)


def fsk_wave(bits, f0: float = 1070.0, f1: float = 1270.0, baud: float = 300.0,
             sr: int = SR) -> np.ndarray:
    """Generate a 2-FSK waveform (e.g. Bell 103) from a bit list."""
    dur = 1.0 / baud
    parts = [_tone(f1 if b else f0, dur, sr) for b in bits]
    return np.concatenate(parts) if parts else _silence(0.1, sr)


def stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack two mono signals into an (n, 2) stereo array (zero-padded)."""
    n = max(len(left), len(right))
    l = np.pad(left, (0, n - len(left)))
    r = np.pad(right, (0, n - len(right)))
    return np.stack([l, r], axis=1)


def write_wav(path: str, data: np.ndarray, sr: int = SR) -> str:
    import soundfile as sf
    sf.write(path, data, sr)
    return path
