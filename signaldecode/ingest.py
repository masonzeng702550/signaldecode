"""Ingest stage: decode any supported input into the unified ``Signal`` model.

A ``Signal`` carries float32 samples in ``[-1, 1]``, the sample rate, and—when
available—the original (pre-downmix) per-channel samples so later stages can run
stereo-difference analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Analysis-time base sample rate. High enough to keep DTMF (up to 1633 Hz) and
# typical SSTV/RTTY tones well below Nyquist, low enough to stay fast.
DEFAULT_ANALYSIS_SR = 44100

SUPPORTED_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".aiff")


@dataclass
class Signal:
    """Unified internal representation of an ingested audio signal.

    Attributes:
        mono: float32 array, shape ``(n,)``, downmixed to mono, range ``[-1, 1]``.
        channels: float32 array, shape ``(c, n)``, original per-channel samples.
        sr: sample rate in Hz.
        path: source file path (informational).
    """

    mono: np.ndarray
    channels: np.ndarray
    sr: int
    path: str

    @property
    def n_channels(self) -> int:
        return int(self.channels.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.mono.shape[0]) / float(self.sr) if self.sr else 0.0

    @property
    def is_stereo(self) -> bool:
        return self.n_channels >= 2


class IngestError(Exception):
    """Raised when an input file cannot be loaded or decoded."""


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Normalise integer/float PCM to float32 in [-1, 1]."""
    if np.issubdtype(data.dtype, np.floating):
        out = data.astype(np.float32)
    else:
        info = np.iinfo(data.dtype)
        # Center unsigned formats around zero, then scale by the larger bound.
        bias = 0.0 if info.min < 0 else (info.max + 1) / 2.0
        scale = float(max(abs(info.min), info.max))
        out = (data.astype(np.float64) - bias) / scale
        out = out.astype(np.float32)
    return np.clip(out, -1.0, 1.0)


def load(path: str, analysis_sr: int = DEFAULT_ANALYSIS_SR) -> Signal:
    """Load ``path`` into a :class:`Signal`, resampled to ``analysis_sr``.

    Tries ``soundfile`` first (wav/flac/ogg), then ``librosa``/``audioread``
    (mp3 and anything ffmpeg can read), then ``scipy.io.wavfile`` as a last
    resort for plain PCM wav.
    """
    if not os.path.exists(path):
        raise IngestError(f"input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    data, sr = _read_any(path, ext)
    # soundfile returns (n,) or (n, c); transpose to (c, n).
    if data.ndim == 1:
        channels = data[np.newaxis, :]
    else:
        channels = data.T
    channels = channels.astype(np.float32, copy=False)

    if sr != analysis_sr:
        channels = _resample(channels, sr, analysis_sr)
        sr = analysis_sr

    mono = channels.mean(axis=0).astype(np.float32)
    return Signal(mono=mono, channels=channels, sr=sr, path=path)


def _read_any(path: str, ext: str):
    errors = []

    # 1) soundfile — best for wav/flac/ogg, returns float in [-1, 1].
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(data), int(sr)
    except Exception as exc:  # pragma: no cover - depends on installed backends
        errors.append(f"soundfile: {exc}")

    # 2) librosa — handles mp3 and exotic formats via audioread/ffmpeg.
    try:
        import librosa

        data, sr = librosa.load(path, sr=None, mono=False)
        return np.asarray(data), int(sr)
    except Exception as exc:  # pragma: no cover
        errors.append(f"librosa: {exc}")

    # 3) scipy.io.wavfile — last resort for plain PCM wav.
    if ext == ".wav":
        try:
            from scipy.io import wavfile

            sr, data = wavfile.read(path)
            return _to_float32(np.asarray(data)), int(sr)
        except Exception as exc:  # pragma: no cover
            errors.append(f"scipy: {exc}")

    raise IngestError(f"could not decode {path!r}: " + "; ".join(errors))


def _resample(channels: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample each channel from ``src_sr`` to ``dst_sr``."""
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(int(src_sr), int(dst_sr))
    up, down = int(dst_sr // g), int(src_sr // g)
    out = np.stack([resample_poly(ch, up, down) for ch in channels])
    return out.astype(np.float32)
