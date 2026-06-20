"""Heuristic classifier.

Maps a :class:`~signaldecode.features.Features` vector to a ranked list of
``Candidate`` types with heuristic confidence scores in ``[0, 1]``. The scores
are deliberately documented as heuristic in the report — they are distances from
threshold, not calibrated probabilities (calibration is future work).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    type: str
    confidence: float
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "confidence": round(self.confidence, 4),
            "params": self.params,
        }


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def _score_morse(f) -> Candidate | None:
    # One dominant tone, clear on/off keying, dot/dash-separable timing.
    if f.tone_count == 0 or f.onoff_transitions < 4:
        return None
    single_tone = 1.0 if f.tone_count == 1 else max(0.0, 1.0 - 0.4 * (f.tone_count - 1))
    keyed = 1.0 if 0.1 <= f.onoff_ratio <= 0.85 else 0.3
    conf = _clip(0.45 * single_tone + 0.2 * keyed + 0.35 * f.keying_regular)
    if conf < 0.2:
        return None
    return Candidate("morse", conf, {"transitions": f.onoff_transitions})


def _score_dtmf(f) -> Candidate | None:
    if not f.dtmf_pair:
        return None
    # Strong, specific signature: a low-group and a high-group tone co-present.
    conf = _clip(0.8 + 0.2 * min(1.0, f.band_occupancy))
    return Candidate("dtmf", conf, {})


def _score_rtty(f) -> Candidate | None:
    if f.tone_count != 2:
        return None
    shift = f.fsk_shift
    common = [170, 425, 450, 850]
    near = min((abs(shift - c) for c in common), default=9999)
    if near > 60:
        return None
    baud_ok = any(abs(f.baud_estimate - b) < 10 for b in (45.45, 50, 75))
    conf = _clip(0.5 * (1.0 - near / 60.0) + (0.3 if baud_ok else 0.0) + 0.2)
    return Candidate("rtty", conf, {"shift_hz": round(shift, 1)})


def _score_afsk(f) -> Candidate | None:
    if f.tone_count != 2:
        return None
    # Bell 103/202 and 1200-baud pairs have wide shifts relative to RTTY.
    shift = f.fsk_shift
    if shift < 150:
        return None
    baud_ok = any(abs(f.baud_estimate - b) < 80 for b in (300, 1200))
    conf = _clip(0.3 + (0.4 if baud_ok else 0.0) + 0.2 * min(1.0, shift / 1000.0))
    if conf < 0.25:
        return None
    return Candidate("afsk", conf, {"shift_hz": round(shift, 1)})


def classify(features, threshold: float = 0.2) -> list[Candidate]:
    """Return candidate signal types ranked by confidence (descending).

    Candidates below ``threshold`` are dropped. ``--deep`` mode lowers the
    threshold via the caller.
    """
    scorers = (_score_morse, _score_dtmf, _score_rtty, _score_afsk)
    cands = [c for c in (s(features) for s in scorers) if c is not None]
    cands = [c for c in cands if c.confidence >= threshold]
    cands.sort(key=lambda c: c.confidence, reverse=True)
    return cands
