"""SignalDecode — unified auto-triage and decode pipeline for CTF audio/signal challenges.

Feed it a ``.wav`` (``.mp3``/``.flac`` via P1) and it classifies the modulation/
encoding present, dispatches to the matching decoder, always runs a set of
"never-skip" analyses (spectrogram, LSB, reverse, stereo diff) and produces a
single aggregated report with confidence scores.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
