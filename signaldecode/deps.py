"""External dependency checks.

Missing tools never abort the pipeline — they downgrade the affected decoder to
``skipped`` and the report explains how to install them.
"""

from __future__ import annotations

import shutil

EXTERNAL_TOOLS = {
    "multimon-ng": "Morse/DTMF/AFSK/POCSAG cross-decode (apt install multimon-ng)",
    "ffmpeg": "mp3 / non-wav decoding (apt install ffmpeg / brew install ffmpeg)",
}


def check() -> dict:
    """Return ``{tool: {"present": bool, "purpose": str}}`` for each external tool."""
    return {
        name: {"present": shutil.which(name) is not None, "purpose": purpose}
        for name, purpose in EXTERNAL_TOOLS.items()
    }


def missing() -> list:
    return [name for name, info in check().items() if not info["present"]]
