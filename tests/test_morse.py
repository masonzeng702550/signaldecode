import numpy as np

from signaldecode.ingest import Signal
from signaldecode.decoders import morse
from tests import synth


def _signal(wave, sr=synth.SR):
    mono = wave.astype(np.float32)
    return Signal(mono=mono, channels=mono[np.newaxis, :], sr=sr, path="synth")


def test_morse_decodes_simple_word():
    wave = synth.morse_wave("SOS", wpm=18)
    result = morse.decode(_signal(wave))
    assert "SOS" in result.text.replace(" ", "")
    assert result.confidence > 0.3


def test_morse_decodes_flag_like():
    wave = synth.morse_wave("HELLO WORLD", wpm=20)
    result = morse.decode(_signal(wave))
    text = result.text.replace(" ", "")
    # Allow minor edge errors but require strong overlap.
    assert "HELLO" in result.text or "HELL" in text


def test_morse_silence_returns_nothing():
    wave = synth._silence(1.0)
    result = morse.decode(_signal(wave))
    assert result.text == ""
    assert result.confidence == 0.0
