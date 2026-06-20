import json
import os

from signaldecode.pipeline import run_pipeline
from tests import synth


def test_end_to_end_dtmf(tmp_path):
    wav = str(tmp_path / "chal.wav")
    synth.write_wav(wav, synth.dtmf_wave("8675309"))
    outdir = str(tmp_path / "report")

    result = run_pipeline(wav, outdir=outdir, use_external=False)

    assert os.path.exists(os.path.join(outdir, "report.md"))
    assert os.path.exists(os.path.join(outdir, "result.json"))
    assert os.path.exists(os.path.join(outdir, "spectrogram.png"))

    with open(os.path.join(outdir, "result.json")) as fh:
        loaded = json.load(fh)
    assert loaded["channels"] == 1
    decoded = " ".join(d["text"] for d in loaded["decodes"])
    assert "8675309" in decoded


def test_end_to_end_morse(tmp_path):
    wav = str(tmp_path / "morse.wav")
    synth.write_wav(wav, synth.morse_wave("SOS", wpm=18))
    outdir = str(tmp_path / "out")
    result = run_pipeline(wav, outdir=outdir, use_external=False)
    decoded = " ".join(d["text"] for d in result["decodes"]).replace(" ", "")
    assert "SOS" in decoded


def test_always_on_runs_even_with_no_candidates(tmp_path):
    # Pure noise: classifier should find nothing, but always-on must still run.
    import numpy as np
    noise = (np.random.RandomState(1).randn(synth.SR) * 0.1).astype("float32")
    wav = str(tmp_path / "noise.wav")
    synth.write_wav(wav, noise)
    outdir = str(tmp_path / "out")
    result = run_pipeline(wav, outdir=outdir, use_external=False)
    assert result["always_on"]["spectrogram"] is not None
    assert os.path.exists(os.path.join(outdir, "spectrogram.png"))
