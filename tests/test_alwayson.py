import os

import numpy as np

from signaldecode.ingest import Signal
from signaldecode import alwayson
from tests import synth


def _signal(wave, sr=synth.SR):
    if wave.ndim == 2:
        channels = wave.T.astype(np.float32)
        mono = channels.mean(axis=0)
    else:
        mono = wave.astype(np.float32)
        channels = mono[np.newaxis, :]
    return Signal(mono=mono.astype(np.float32), channels=channels, sr=sr, path="synth")


def test_lsb_extracts_hidden_flag(tmp_path):
    # Embed "flag{lsb_test}" into the LSBs of an int16 carrier.
    payload = b"....flag{lsb_test}...."
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    carrier = (np.random.RandomState(0).randint(-1000, 1000, size=len(bits) + 100)
               ).astype(np.int16)
    carrier[: len(bits)] = (carrier[: len(bits)] & ~1) | bits
    sig = _signal((carrier.astype(np.float32) / 32768.0))

    res = alwayson.run(sig, str(tmp_path), original_int_samples=carrier)
    joined = " ".join(res.lsb_printable_hits)
    assert "flag{lsb_test}" in joined
    assert os.path.exists(os.path.join(tmp_path, "lsb_extract.bin"))


def test_reverse_and_stereo_written(tmp_path):
    left = synth.dtmf_wave("12")
    right = synth._silence(len(left) / synth.SR)
    sig = _signal(synth.stereo(left, right))
    res = alwayson.run(sig, str(tmp_path))
    assert res.reversed_wav and os.path.exists(os.path.join(tmp_path, res.reversed_wav))
    assert res.stereo_diff_wav and os.path.exists(
        os.path.join(tmp_path, res.stereo_diff_wav))


def test_spectrogram_generated(tmp_path):
    sig = _signal(synth.morse_wave("E"))
    res = alwayson.run(sig, str(tmp_path))
    # matplotlib is a declared dependency; spectrogram should be produced.
    assert res.spectrogram is not None
    assert os.path.exists(os.path.join(tmp_path, res.spectrogram))
