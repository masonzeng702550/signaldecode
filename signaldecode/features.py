"""Feature extraction.

All features are derived from the mono ``Signal`` with numpy/scipy only. The
output ``Features`` object feeds the heuristic classifier and is serialised into
``result.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.signal import spectrogram, find_peaks

# DTMF tone groups (Hz) used by both feature extraction and the classifier.
DTMF_LOW = (697, 770, 852, 941)
DTMF_HIGH = (1209, 1336, 1477, 1633)


@dataclass
class Features:
    spectral_centroid: float = 0.0
    dominant_tones: list = field(default_factory=list)  # [(freq_hz, rel_power)]
    tone_count: int = 0
    onoff_ratio: float = 0.0          # fraction of time keyed "on"
    onoff_transitions: int = 0        # number of on<->off edges
    keying_regular: float = 0.0       # 0..1 how Morse-like the on/off timing is
    band_occupancy: float = 0.0       # fraction of spectral energy in main band
    fsk_shift: float = 0.0            # Hz between two dominant peaks (0 if not 2)
    baud_estimate: float = 0.0        # symbol rate estimate (Hz)
    stereo_diff_energy: float = 0.0   # L-R residual energy ratio (0 if mono)
    dtmf_pair: bool = False           # one low + one high DTMF tone co-present

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dominant_tones"] = [
            {"freq": round(f, 2), "power": round(p, 4)} for f, p in self.dominant_tones
        ]
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _power_spectrum(x: np.ndarray, sr: int):
    n = int(2 ** np.ceil(np.log2(max(len(x), 2))))
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win, n=n)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return freqs, spec


def _spectral_centroid(freqs, spec) -> float:
    total = spec.sum()
    if total <= 0:
        return 0.0
    return float((freqs * spec).sum() / total)


def _dominant_tones(freqs, spec, max_tones=6, rel_thresh=0.15):
    if spec.max() <= 0:
        return []
    norm = spec / spec.max()
    # Require peaks separated by ~30 Hz so we don't split one tone into many.
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    distance = max(1, int(round(30.0 / df)))
    peaks, props = find_peaks(norm, height=rel_thresh, distance=distance)
    if len(peaks) == 0:
        return []
    order = np.argsort(props["peak_heights"])[::-1][:max_tones]
    sel = peaks[order]
    return [(float(freqs[i]), float(norm[i])) for i in sel]


def _onoff_envelope(x: np.ndarray, sr: int):
    """Return (on_fraction, transitions, regularity) from the energy envelope."""
    # ~5 ms frames; smooth, then threshold at a fraction of the peak.
    frame = max(1, int(sr * 0.005))
    env = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(frame) / frame, "same"))
    if env.max() <= 0:
        return 0.0, 0, 0.0
    env /= env.max()
    keyed = env > 0.25
    on_fraction = float(keyed.mean())
    edges = np.diff(keyed.astype(np.int8))
    transitions = int(np.count_nonzero(edges))

    # Regularity: run lengths of "on" segments should cluster into dot/dash.
    runs = _run_lengths(keyed)
    on_runs = [r for val, r in runs if val]
    regularity = _dotdash_regularity(np.array(on_runs, dtype=float)) if on_runs else 0.0
    return on_fraction, transitions, regularity


def _run_lengths(mask: np.ndarray):
    if len(mask) == 0:
        return []
    runs = []
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


def _dotdash_regularity(runs: np.ndarray) -> float:
    """How well a set of "on" run lengths separates into two clusters (dot/dash).

    Returns 0..1; high when lengths cluster tightly around a short and a long
    value at roughly a 1:3 ratio (Morse), low when uniformly spread (noise).
    """
    if len(runs) < 2:
        return 0.0
    short = np.percentile(runs, 25)
    longv = np.percentile(runs, 75)
    if short <= 0:
        return 0.0
    ratio = longv / short
    # Reward ratios near the Morse 1:3 ideal; tolerate 1.8..4.5.
    ratio_score = max(0.0, 1.0 - abs(ratio - 3.0) / 3.0)
    # Reward bimodality: most runs near short or near long, few in between.
    mid = (short + longv) / 2.0
    near = np.mean((runs < mid * 0.8) | (runs > mid * 1.2))
    return float(np.clip(0.5 * ratio_score + 0.5 * near, 0.0, 1.0))


def _band_occupancy(freqs, spec) -> float:
    total = spec.sum()
    if total <= 0:
        return 0.0
    peak = int(np.argmax(spec))
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    half = max(1, int(round(150.0 / df)))  # +/- 150 Hz around the dominant peak
    lo, hi = max(0, peak - half), min(len(spec), peak + half + 1)
    return float(spec[lo:hi].sum() / total)


def _baud_estimate(x: np.ndarray, sr: int) -> float:
    """Estimate symbol rate from the autocorrelation of the energy envelope."""
    frame = max(1, int(sr * 0.001))
    env = np.convolve(x.astype(np.float64) ** 2, np.ones(frame) / frame, "same")
    env -= env.mean()
    if np.allclose(env, 0):
        return 0.0
    ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
    # Look for the first strong peak after lag 0 within 10..2000 baud.
    lo = max(1, int(sr / 2000.0))
    hi = min(len(ac) - 1, int(sr / 10.0))
    if hi <= lo:
        return 0.0
    seg = ac[lo:hi]
    if seg.max() <= 0:
        return 0.0
    lag = lo + int(np.argmax(seg))
    return float(sr / lag) if lag else 0.0


def _stereo_diff_energy(channels: np.ndarray) -> float:
    if channels.shape[0] < 2:
        return 0.0
    left, right = channels[0], channels[1]
    n = min(len(left), len(right))
    diff = left[:n] - right[:n]
    summ = left[:n] + right[:n]
    e_sum = float(np.sum(summ ** 2)) + 1e-12
    e_diff = float(np.sum(diff ** 2))
    return e_diff / e_sum


def _dtmf_pair(tones, tol=25.0) -> bool:
    low = any(any(abs(f - t) <= tol for t in DTMF_LOW) for f, _ in tones)
    high = any(any(abs(f - t) <= tol for t in DTMF_HIGH) for f, _ in tones)
    return bool(low and high)


def extract(signal) -> Features:
    """Extract the full :class:`Features` vector from a ``Signal``."""
    x = signal.mono
    sr = signal.sr
    freqs, spec = _power_spectrum(x, sr)

    tones = _dominant_tones(freqs, spec)
    on_frac, transitions, regularity = _onoff_envelope(x, sr)

    fsk_shift = 0.0
    if len(tones) >= 2:
        f0, f1 = sorted(f for f, _ in tones[:2])
        fsk_shift = float(f1 - f0)

    return Features(
        spectral_centroid=_spectral_centroid(freqs, spec),
        dominant_tones=tones,
        tone_count=len(tones),
        onoff_ratio=on_frac,
        onoff_transitions=transitions,
        keying_regular=regularity,
        band_occupancy=_band_occupancy(freqs, spec),
        fsk_shift=fsk_shift,
        baud_estimate=_baud_estimate(x, sr),
        stereo_diff_energy=_stereo_diff_energy(signal.channels),
        dtmf_pair=_dtmf_pair(tones),
    )
