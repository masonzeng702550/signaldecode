"""Report stage: aggregate everything into ``report.md`` + ``result.json``.

Presentation rules: spectrogram first, candidates sorted by confidence
descending, every item labelled with its score, and ``flag{...}``-shaped strings
highlighted.
"""

from __future__ import annotations

import json
import os
import re

_FLAG_RE = re.compile(r"((?:flag|ctf|key)\{[^}]{1,200}\})", re.I)


def _highlight_flags(text: str) -> str:
    return _FLAG_RE.sub(r"**`\1`**", text)


def build_result(signal, features, candidates, decodes, always_on, warnings) -> dict:
    return {
        "tool": "signaldecode",
        "input": os.path.basename(signal.path),
        "duration_s": round(signal.duration_s, 3),
        "sample_rate": signal.sr,
        "channels": signal.n_channels,
        "features": features.to_dict(),
        "candidates": [c.to_dict() for c in candidates],
        "decodes": [d.to_dict() for d in decodes],
        "always_on": always_on.to_dict(),
        "warnings": warnings,
    }


def write(result: dict, outdir: str, fmt: str = "md") -> str:
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, "result.json")
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    md = _render_markdown(result)
    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w") as fh:
        fh.write(md)
    return md_path


def _render_markdown(r: dict) -> str:
    L = []
    L.append(f"# SignalDecode report — `{r['input']}`\n")
    L.append(
        f"- Duration: **{r['duration_s']} s** · Sample rate: **{r['sample_rate']} Hz** "
        f"· Channels: **{r['channels']}**\n"
    )

    # 1) Spectrogram first — the anti-miss core.
    ao = r["always_on"]
    L.append("## Spectrogram (always-on)\n")
    if ao.get("spectrogram"):
        L.append(f"![spectrogram]({ao['spectrogram']})\n")
        if ao.get("spectrogram_log"):
            L.append(f"![spectrogram log]({ao['spectrogram_log']})\n")
        L.append("> Inspect the spectrogram for hidden text even when no decoder fired.\n")
    else:
        L.append("_Spectrogram not generated (matplotlib unavailable)._\n")

    # 2) Candidates, highest confidence first.
    L.append("## Classification candidates\n")
    if r["candidates"]:
        L.append("| type | confidence | params |")
        L.append("|------|-----------|--------|")
        for c in r["candidates"]:
            L.append(f"| `{c['type']}` | {c['confidence']:.2f} | `{c['params']}` |")
        L.append("")
    else:
        L.append("_No known modulation detected. Review the spectrogram and "
                 "always-on artifacts below._\n")

    # 3) Decodes.
    L.append("## Decode results\n")
    if r["decodes"]:
        for d in r["decodes"]:
            status = "⏭️ skipped" if d["skipped"] else f"confidence {d['confidence']:.2f}"
            L.append(f"### `{d['type']}` via {d['decoder']} — {status}")
            if d["note"]:
                L.append(f"_{d['note']}_")
            if d["text"]:
                L.append(f"\n```\n{d['text']}\n```")
                hl = _highlight_flags(d["text"])
                if hl != d["text"]:
                    L.append(f"\nFlag-shaped match: {hl}")
            L.append("")
    else:
        L.append("_No decoders ran._\n")

    # 4) Always-on artifacts.
    L.append("## Always-on artifacts\n")
    lsb = ao.get("lsb", {})
    if lsb.get("printable_hits") or lsb.get("magic_hits"):
        L.append("**LSB extraction hits:**")
        for h in lsb.get("printable_hits", [])[:10]:
            L.append(f"- `{_highlight_flags(h)}`")
        if lsb.get("magic_hits"):
            L.append(f"- file magics: {', '.join(lsb['magic_hits'])}")
        L.append("")
    if ao.get("reversed"):
        L.append(f"- Reversed audio: `{ao['reversed']}`")
    if ao.get("stereo_diff"):
        L.append(f"- Stereo difference (L−R): `{ao['stereo_diff']}`")
    if lsb.get("artifact"):
        L.append(f"- LSB raw bytes: `{lsb['artifact']}`")
    L.append("")

    # 5) Warnings.
    if r["warnings"]:
        L.append("## Warnings\n")
        for w in r["warnings"]:
            L.append(f"- {w}")
        L.append("")

    L.append("---")
    L.append("_Confidence scores are heuristic, not calibrated probabilities._")
    return "\n".join(L) + "\n"
