"""Shared decoder result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DecodeResult:
    """Output of a single decoder.

    Attributes:
        type: signal type, e.g. ``"morse"``.
        decoder: which engine produced this, e.g. ``"builtin"`` or ``"multimon-ng"``.
        confidence: heuristic confidence of this decode in ``[0, 1]``.
        text: decoded text (may be empty).
        artifact: relative path to a written artifact, if any.
        skipped: True when the decoder could not run (e.g. missing tool).
        note: human-readable status / reason.
    """

    type: str
    decoder: str
    confidence: float = 0.0
    text: str = ""
    artifact: Optional[str] = None
    skipped: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "decoder": self.decoder,
            "confidence": round(self.confidence, 4),
            "text": self.text,
            "artifact": self.artifact,
            "skipped": self.skipped,
            "note": self.note,
        }
