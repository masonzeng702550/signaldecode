import numpy as np

from signaldecode.ingest import Signal
from signaldecode import features as feat
from signaldecode import classify as clf
from tests import synth


def _signal(wave, sr=synth.SR):
    if wave.ndim == 2:
        channels = wave.T.astype(np.float32)
        mono = channels.mean(axis=0)
    else:
        mono = wave.astype(np.float32)
        channels = mono[np.newaxis, :]
    return Signal(mono=mono.astype(np.float32), channels=channels, sr=sr, path="synth")


def test_dtmf_features_and_classification():
    sig = _signal(synth.dtmf_wave("159"))
    f = feat.extract(sig)
    assert f.dtmf_pair is True
    cands = clf.classify(f)
    assert cands and cands[0].type == "dtmf"


def test_morse_classified_as_morse():
    sig = _signal(synth.morse_wave("PARIS", wpm=18))
    f = feat.extract(sig)
    cands = clf.classify(f)
    types = [c.type for c in cands]
    assert "morse" in types


def test_stereo_diff_energy_detected():
    left = synth.morse_wave("E", wpm=18)
    right = synth._silence(len(left) / synth.SR)
    sig = _signal(synth.stereo(left, right))
    f = feat.extract(sig)
    assert f.stereo_diff_energy > 0.1


def test_no_signal_low_confidence():
    sig = _signal(synth._silence(1.0))
    f = feat.extract(sig)
    cands = clf.classify(f)
    assert cands == []
