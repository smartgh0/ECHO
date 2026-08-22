#!/usr/bin/env python3
"""Shared generation stop rules to prevent training-format bleed."""

from __future__ import annotations

import re

# Hard turn boundaries from the training transcript format.
STOP_MARKERS = (
    "\nuser:",
    "\nuser :",
    "\necho:",
    "\necho :",
    "\ntool_result",
    "\ntool_result:",
)

# Mid-line bleed (model continues as if a new turn without newline).
INLINE_BLEED = re.compile(
    r"(?<![A-Za-z0-9_])(?:user\s*:|echo\s*:|tool_result\b)",
    re.IGNORECASE,
)


def truncate_format_bleed(text: str, *, stop_after_first_tool: bool = False) -> str:
    """Cut generated text at the first hallucinated next-turn / tool_result marker."""
    cut = len(text)
    for marker in STOP_MARKERS:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)

    # After the model has produced some content, also cut mid-line role labels.
    # Skip the very start so a legitimate leading token isn't over-penalized.
    search_from = min(8, len(text))
    match = INLINE_BLEED.search(text, search_from)
    if match:
        cut = min(cut, match.start())

    if stop_after_first_tool:
        # One-shot agent policy: first tool line, then stop (no fake tool_result).
        tool_idx = text.find("tool:")
        if tool_idx != -1:
            nl = text.find("\n", tool_idx)
            if nl != -1:
                cut = min(cut, nl)

    return text[:cut].rstrip()
