"""Command-line interface.

    signaldecode INPUT [-o OUTDIR] [--deep] [--only TYPES] [--skip TYPES]
                       [--format md|html] [--no-external] [-v]
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .ingest import IngestError
from .pipeline import run_pipeline


def _parse_types(value: str | None) -> set | None:
    if not value:
        return None
    return {t.strip().lower() for t in value.split(",") if t.strip()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="signaldecode",
        description="Unified auto-triage and decode pipeline for CTF audio/signal challenges.",
    )
    p.add_argument("input", help="input file (.wav/.flac/.mp3)")
    p.add_argument("-o", "--output", default="report", metavar="OUTDIR",
                   help="output directory for report and artifacts (default: ./report/)")
    p.add_argument("--deep", action="store_true",
                   help="deep scan: lower thresholds, run more decoders")
    p.add_argument("--only", metavar="TYPES",
                   help="only run these comma-separated types (e.g. morse,dtmf)")
    p.add_argument("--skip", metavar="TYPES",
                   help="skip these comma-separated types")
    p.add_argument("--format", choices=["md", "html"], default="md",
                   help="report format (md; html reserved for P1)")
    p.add_argument("--no-external", action="store_true",
                   help="do not call external tools (offline / CI friendly)")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    p.add_argument("--version", action="version", version=f"signaldecode {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_pipeline(
            args.input,
            outdir=args.output,
            deep=args.deep,
            only=_parse_types(args.only),
            skip=_parse_types(args.skip),
            use_external=not args.no_external,
            fmt=args.format,
            verbose=args.verbose,
        )
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - report unexpected failures cleanly
        print(f"internal error: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, args.output)
    return 0


def _print_summary(result: dict, outdir: str) -> None:
    print(f"\nSignalDecode — {result['input']} "
          f"({result['duration_s']}s, {result['channels']}ch)")
    if result["candidates"]:
        top = result["candidates"][0]
        print(f"  top candidate: {top['type']} (confidence {top['confidence']:.2f})")
    else:
        print("  no known modulation detected — check the spectrogram")
    for d in result["decodes"]:
        if d["text"]:
            print(f"  [{d['type']}/{d['decoder']}] {d['text'][:80]}")
    print(f"  report: {outdir}/report.md")


if __name__ == "__main__":
    raise SystemExit(main())
