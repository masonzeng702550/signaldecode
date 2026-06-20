import numpy as np

from signaldecode.ingest import Signal
from signaldecode.decoders import dtmf
from tests import synth


def _signal(wave, sr=synth.SR):
    mono = wave.astype(np.float32)
    return Signal(mono=mono, channels=mono[np.newaxis, :], sr=sr, path="synth")


def test_dtmf_decodes_sequence():
    wave = synth.dtmf_wave("1234567890")
    result = dtmf.decode(_signal(wave))
    assert result.text == "1234567890"
    assert result.confidence > 0.5


def test_dtmf_decodes_with_repeats():
    wave = synth.dtmf_wave("112233")
    result = dtmf.decode(_signal(wave))
    assert result.text == "112233"


def test_dtmf_silence():
    wave = synth._silence(0.5)
    result = dtmf.decode(_signal(wave))
    assert result.text == ""
