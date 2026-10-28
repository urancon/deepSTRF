"""Leak-aware train / val / test split for Gosselin 2025."""
from __future__ import annotations

from typing import List, Sequence, Tuple


def split_units(stim_meta: Sequence[dict], gain_copies: Sequence[Sequence[str]]
                ) -> Tuple[List[str], List[str]]:
    names = [m["name"] for m in stim_meta]
    return names, [m["category"] for m in stim_meta]
