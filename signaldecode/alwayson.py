"""Always-on analyses — run regardless of classification.

These are the anti-miss core: a flag is often hidden in the spectrogram, in the
audio LSBs, in a reversed copy, or in a hidden second channel. Their outputs are
*always* written to the report.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

# Common file-header magics worth flagging in an LSB byte stream.
_MAGICS = {
    b"PK\x03\x04": "zip",
    b"\x89PNG": "png",
    b"\xff\xd8\xff": "jpeg",
    b"GIF8": "gif",
    b"%PDF": "pdf",
    b"\x7fELF": "elf",
    b"ID3": "mp3",
    b"RIFF": "riff/wav",
}
_PRINTABLE_RE = re.compile(rb"[ -~]{4,}")
_FLAG_RE = re.compile(rb"(?:flag|ctf|key)\{[^}]{1,200}\}", re.I)


@dataclass
class AlwaysOnResult:
    spectrogram: str | None = None
    spectrogram_log: str | None = None
    lsb_artifact: str | None = None
    lsb_printable_hits: list = field(default_factory=list)
    lsb_magic_hits: list = field(default_factory=list)
    reversed_wav: str | None = None
    stereo_diff_wav: str | None = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "spectrogram": self.spectrogram,
            "spectrogram_log": self.spectrogram_log,
            "lsb": {
                "artifact": self.lsb_artifact,
                "printable_hits": self.lsb_printable_hits,
                "magic_hits": self.lsb_magic_hits,
            },
            "reversed": self.reversed_wav,
            "stereo_diff": self.stereo_diff_wav,
            "notes": self.notes,
        }


def run(signal, outdir: str, *, original_int_samples=None) -> AlwaysOnResult:
    res = AlwaysOnResult()
    _spectrograms(signal, outdir, res)
    _lsb(signal, outdir, res, original_int_samples)
    _reverse(signal, outdir, res)
    _stereo_diff(signal, outdir, res)
    return res


def _spectrograms(signal, outdir, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.signal import spectrogram
    except Exception as exc:  # pragma: no cover - matplotlib optional at runtime
        res.notes.append(f"spectrogram skipped: {exc}")
        return

    x = signal.mono
    sr = signal.sr
    nper = min(2048, max(256, len(x) // 100 or 256))
    f, t, Sxx = spectrogram(x, fs=sr, nperseg=nper, noverlap=nper // 2)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    lin = os.path.join(outdir, "spectrogram.png")
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    plt.ylabel("Frequency (Hz)"); plt.xlabel("Time (s)")
    plt.title("Spectrogram (linear frequency)")
    plt.colorbar(label="dB"); plt.tight_layout()
    plt.savefig(lin, dpi=120); plt.close()
    res.spectrogram = os.path.basename(lin)

    log = os.path.join(outdir, "spectrogram_log.png")
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    plt.yscale("symlog"); plt.ylabel("Frequency (Hz, log)"); plt.xlabel("Time (s)")
    plt.title("Spectrogram (log frequency)")
    plt.colorbar(label="dB"); plt.tight_layout()
    plt.savefig(log, dpi=120); plt.close()
    res.spectrogram_log = os.path.basename(log)


def _lsb(signal, outdir, res, original_int_samples):
    # Prefer the original integer PCM if available; else quantise float to int16.
    if original_int_samples is not None:
        ints = np.asarray(original_int_samples).astype(np.int64).ravel()
    else:
        ints = (np.clip(signal.mono, -1, 1) * 32767.0).astype(np.int16).astype(np.int64)
    if ints.size == 0:
        res.notes.append("lsb skipped: no samples")
        return
    bits = (ints & 1).astype(np.uint8)
    nbytes = bits.size // 8
    if nbytes == 0:
        res.notes.append("lsb skipped: too few samples")
        return
    byte_arr = np.packbits(bits[: nbytes * 8])
    raw = byte_arr.tobytes()

    path = os.path.join(outdir, "lsb_extract.bin")
    with open(path, "wb") as fh:
        fh.write(raw)
    res.lsb_artifact = os.path.basename(path)

    hits = [m.group(0).decode("ascii", "replace") for m in _PRINTABLE_RE.finditer(raw)]
    flags = [m.group(0).decode("ascii", "replace") for m in _FLAG_RE.finditer(raw)]
    # Surface flag-shaped strings first, then the longest printable runs.
    res.lsb_printable_hits = flags + sorted(set(hits) - set(flags),
                                            key=len, reverse=True)[:20]
    for magic, name in _MAGICS.items():
        if magic in raw:
            res.lsb_magic_hits.append(name)


def _reverse(signal, outdir, res):
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        res.notes.append(f"reversed wav skipped: {exc}")
        return
    rev = signal.mono[::-1]
    path = os.path.join(outdir, "reversed.wav")
    sf.write(path, rev, signal.sr)
    res.reversed_wav = os.path.basename(path)


def _stereo_diff(signal, outdir, res):
    if not signal.is_stereo:
        return
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        res.notes.append(f"stereo diff skipped: {exc}")
        return
    left, right = signal.channels[0], signal.channels[1]
    n = min(len(left), len(right))
    diff = left[:n] - right[:n]
    path = os.path.join(outdir, "stereo_diff.wav")
    sf.write(path, diff, signal.sr)
    res.stereo_diff_wav = os.path.basename(path)
