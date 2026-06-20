# SignalDecode

**Unified auto-triage and decode pipeline for CTF audio/signal challenges — think of it as the audio version of Aperi'Solve.**

Feed it a `.wav` (`.mp3`/`.flac` via ffmpeg) and it:

1. **Classifies** which modulation/encoding is present and gives a confidence score.
2. **Dispatches** to the matching decoder (Morse/CW, DTMF today; RTTY/AFSK/SSTV planned).
3. **Always runs** a set of never-skip analyses — spectrogram render, audio-LSB extraction, time-reverse, stereo-difference — so flags hidden in the "you should always check this" steps don't slip through.
4. **Aggregates** everything into a single report (`report.md` + machine-readable `result.json`) with the spectrogram up front and `flag{...}`-shaped strings highlighted.

Players spend their time *solving* instead of *trying tools one by one*.

---

## Why

CTF misc/forensics audio challenges hide flags as Morse, DTMF, SSTV, RTTY/AFSK, spectrogram text, audio LSB stego, or simple tricks (reversed audio, a second channel). The usual workflow is manual guess-and-check across Audacity, multimon-ng, QSSTV… Each of those tools decodes *one* thing and assumes you already classified the signal.

SignalDecode fills the exact gap: **classify → auto-dispatch → aggregate report**, covering audio/signal formats. It doesn't reimplement mature decoders — it *orchestrates* them (e.g. wrapping `multimon-ng`).

---

## Install

```bash
pip install -e .
```

Core dependencies: `numpy`, `scipy`, `matplotlib`, `soundfile`.

Optional external tools (auto-detected; missing ones are reported as `skipped`, never fatal):

- `multimon-ng` — cross-decode for Morse/DTMF/AFSK/POCSAG (`apt install multimon-ng`)
- `ffmpeg` — `.mp3` and non-wav decoding (`brew install ffmpeg` / `apt install ffmpeg`)

---

## Usage

```bash
signaldecode INPUT [-o OUTDIR] [--deep] [--only TYPES] [--skip TYPES]
                   [--format md|html] [--no-external] [-v]
```

Examples:

```bash
signaldecode chal.wav -o report/
signaldecode chal.wav -o report/ --deep
signaldecode capture.wav --only morse,dtmf --no-external
```

| Flag | Meaning |
|------|---------|
| `-o, --output` | Output directory (default `./report/`) |
| `--deep` | Lower thresholds, run more decoders |
| `--only TYPES` | Only run these comma-separated types |
| `--skip TYPES` | Skip these types |
| `--no-external` | Don't call external tools (offline / CI) |
| `-v` | Verbose logging |

Exit code is `0` on a completed run (whether or not anything decoded), non-zero only when the input can't be loaded or an internal error occurs.

---

## Output

```
report/
├── report.md            # main report (spectrogram first, candidates by confidence)
├── result.json          # machine-readable summary
├── spectrogram.png      # always-on, linear frequency
├── spectrogram_log.png  # always-on, log frequency
├── lsb_extract.bin      # always-on LSB byte stream
├── reversed.wav         # always-on time-reversed audio
└── stereo_diff.wav      # always-on L−R residual (stereo inputs)
```

---

## Architecture

Five stages — `ingest → classify → dispatch → decode → report` — with **always-on** analyses running alongside classification and *always* written to the report (the anti-miss core).

| Module | Role |
|--------|------|
| `signaldecode/ingest.py` | Decode any input into the unified `Signal` model (float32, resampled) |
| `signaldecode/features.py` | Spectral / on-off / FSK / baud / stereo features |
| `signaldecode/classify.py` | Heuristic classifier → ranked candidates with confidence |
| `signaldecode/dispatch.py` | Build and run the decoder work list |
| `signaldecode/decoders/` | `morse`, `dtmf` (built-in) + `multimon` wrapper |
| `signaldecode/alwayson.py` | Spectrogram, LSB, reverse, stereo-diff |
| `signaldecode/report.py` | `report.md` + `result.json` |
| `signaldecode/pipeline.py` | Orchestrates the whole run |

Modules are loosely coupled: every decoder and always-on analysis is independently callable.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests use **synthetic** signals (`tests/synth.py`) — known Morse/DTMF/FSK waveforms — so the suite is deterministic and needs no external corpus.

---

## Status & roadmap

**MVP (done):** wav ingest, feature extraction, heuristic classifier, built-in Morse/DTMF + multimon-ng wrapper, all always-on analyses, Markdown + JSON report, CLI.

**Planned:** mp3/flac, SSTV/RTTY/AFSK/PSK31 decoders, speed/resample retry branches, HTML report, `--deep` expansion, confidence calibration, IQ input, web UI, plugin API.

---

## Scope & ethics

For CTF and lawful learning only. SignalDecode does not target real protected communications. Confidence scores are heuristic, not calibrated probabilities. Image stego is deliberately out of scope (that's Aperi'Solve's domain); SignalDecode only hands off at the spectrogram level.

## License

MIT — see [LICENSE](LICENSE).
