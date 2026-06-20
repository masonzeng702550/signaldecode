"""Top-level pipeline orchestration: ingest -> classify(+always-on) -> dispatch -> report."""

from __future__ import annotations

import os

from . import ingest, features as feat, classify as clf, dispatch, alwayson, report, deps


def run_pipeline(
    input_path: str,
    outdir: str = "report",
    *,
    deep: bool = False,
    only: set | None = None,
    skip: set | None = None,
    use_external: bool = True,
    fmt: str = "md",
    verbose: bool = False,
) -> dict:
    """Run the full pipeline and write the report. Returns the result dict."""
    warnings: list[str] = []

    def log(msg):
        if verbose:
            print(f"[signaldecode] {msg}")

    os.makedirs(outdir, exist_ok=True)

    log(f"ingest {input_path}")
    signal = ingest.load(input_path)

    log("extract features")
    features = feat.extract(signal)

    threshold = 0.1 if deep else 0.2
    log(f"classify (threshold={threshold})")
    candidates = clf.classify(features, threshold=threshold)

    # Always-on runs regardless of classification (anti-miss core).
    log("always-on analyses")
    ao = alwayson.run(signal, outdir)
    warnings.extend(ao.notes)

    # If LSB found a flag-shaped string, surface it as a candidate too.
    if ao.lsb_printable_hits and any("{" in h and "}" in h for h in ao.lsb_printable_hits):
        candidates.append(clf.Candidate("audio_lsb", 0.7, {"source": "always-on lsb"}))
        candidates.sort(key=lambda c: c.confidence, reverse=True)

    # External tool status -> warnings.
    for name in deps.missing():
        warnings.append(f"external tool not found: {name} "
                        f"({deps.EXTERNAL_TOOLS[name]})")

    log(f"dispatch decoders ({len(candidates)} candidates)")
    decodes = dispatch.run(
        signal, candidates,
        use_external=use_external, only=only, skip=skip,
    )

    log("build report")
    result = report.build_result(signal, features, candidates, decodes, ao, warnings)
    path = report.write(result, outdir, fmt=fmt)
    log(f"report written to {path}")
    return result
