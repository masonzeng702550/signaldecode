"""Dispatch stage: turn classifier candidates into a decode work list and run it.

Built-in decoders run for any candidate above threshold. When external tools are
permitted and present, multimon-ng runs alongside as cross-validation. Built-in
and external results for the same type are kept side by side in the report.
"""

from __future__ import annotations

from .decoders import morse as morse_dec
from .decoders import dtmf as dtmf_dec
from .decoders import multimon
from .decoders.base import DecodeResult

# Types we have a built-in decoder for.
BUILTIN = {"morse", "dtmf"}


def run(signal, candidates, *, use_external: bool = True,
        only: set | None = None, skip: set | None = None) -> list[DecodeResult]:
    """Execute decoders for the given candidates; return all results."""
    results: list[DecodeResult] = []
    seen = set()
    for cand in candidates:
        t = cand.type
        if only and t not in only:
            continue
        if skip and t in skip:
            continue
        if t in seen:
            continue
        seen.add(t)

        if t == "morse":
            results.append(morse_dec.decode(signal))
        elif t == "dtmf":
            results.append(dtmf_dec.decode(signal))

        # External cross-check / coverage for types multimon-ng handles.
        if use_external and t in multimon.DEMOD:
            results.append(multimon.decode(signal, t))

    return results
