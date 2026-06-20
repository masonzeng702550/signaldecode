"""multimon-ng wrapper.

Cross-validates built-in decoders and covers AFSK/POCSAG/etc. It feeds
multimon-ng the raw format it expects (16-bit signed, 22050 Hz, mono) over a
temporary file and parses stdout. If multimon-ng is not on ``PATH`` the result
is marked ``skipped`` rather than raising — the pipeline must never break on a
missing external tool.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.signal import resample_poly

from .base import DecodeResult

MULTIMON_SR = 22050
# multimon-ng demodulator names keyed by our signal type.
DEMOD = {
    "morse": "MORSE_CW",
    "dtmf": "DTMF",
    "afsk": "AFSK1200",
    "pocsag": "POCSAG512",
}

# Lines multimon-ng prints that are status, not decoded payload.
_STATUS_RE = re.compile(r"^(Enabled demodulators|multimon-ng|Available demodulators|-)", re.I)


def available() -> bool:
    return shutil.which("multimon-ng") is not None


def _to_raw_s16(signal) -> bytes:
    x = signal.mono
    if signal.sr != MULTIMON_SR:
        from math import gcd

        g = gcd(int(signal.sr), MULTIMON_SR)
        x = resample_poly(x, MULTIMON_SR // g, signal.sr // g)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def decode(signal, sig_type: str) -> DecodeResult:
    demod = DEMOD.get(sig_type)
    if demod is None:
        return DecodeResult(sig_type, "multimon-ng", skipped=True,
                            note=f"no multimon-ng demod for {sig_type}")
    if not available():
        return DecodeResult(sig_type, "multimon-ng", skipped=True,
                            note="multimon-ng not installed (apt install multimon-ng)")

    raw = _to_raw_s16(signal)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as fh:
            fh.write(raw)
            tmp = fh.name
        proc = subprocess.run(
            ["multimon-ng", "-t", "raw", "-a", demod, "-q", tmp],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return DecodeResult(sig_type, "multimon-ng", skipped=True, note=f"error: {exc}")
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    payload = [ln for ln in lines if not _STATUS_RE.match(ln)]
    text = _clean(payload, demod)
    conf = 0.6 if text else 0.0
    note = "decoded" if text else "no output"
    return DecodeResult(sig_type, "multimon-ng", conf, text, note=note)


def _clean(lines, demod: str) -> str:
    out = []
    for ln in lines:
        # multimon-ng prefixes like "DTMF: 1 2 3" or "MORSE: HELLO".
        m = re.match(r"^[A-Z0-9_]+:\s*(.*)$", ln)
        out.append(m.group(1) if m else ln)
    text = " ".join(p for p in out if p).strip()
    if demod == "DTMF":
        text = text.replace(" ", "")
    return text
