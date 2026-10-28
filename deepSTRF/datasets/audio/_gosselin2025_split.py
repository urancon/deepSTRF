"""Leak-aware train / val / test split for Gosselin 2025.

Kept together (one unit): (1) sounds that are the same waveform up to a gain
(the 50 / 70 dB copies, from ``_gosselin2025_stim_groups.json``); (2) a
decomposed clip's forward original(s), all its components and its
reconstruction. Everything else, including time-reversed clips and up/down
ramps and sweeps, is its own unit. Units are drawn per stratum (the CSV
category, or ``decomposition``) to the set furthest below its target share.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

KEYS = ("train", "val", "test")


def split_units(stim_meta: Sequence[dict], gain_copies: Sequence[Sequence[str]]
                ) -> Tuple[List[str], List[str]]:
    """Per stimulus: the unit it is drawn with, and that unit's stratum."""
    rep = {}
    for g in gain_copies:
        for n in g:
            rep[n] = sorted(g)[0]
    units, strata = [], []
    for m in stim_meta:
        if m["decomposition_role"] is not None:
            units.append(f"decomp:{m['source_clip']}")
            strata.append("decomposition")
        else:
            units.append(rep.get(m["name"], m["name"]))
            strata.append(m["category"])
    return units, strata


def draw_split(units: Sequence[str], strata: Sequence[str], seed, fractions: Sequence[float]
               ) -> Dict[str, List[int]]:
    """Assign whole units to train / val / test, stratum by stratum; each unit,
    in a seeded shuffle, goes to the set furthest below its target share."""
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be 3 shares summing to 1, got {fractions}")
    rng = random.Random(seed)
    size: Dict[str, int] = {}
    stratum_of: Dict[str, str] = {}
    for u, st in zip(units, strata):
        size[u] = size.get(u, 0) + 1
        stratum_of[u] = st
    assign: Dict[str, str] = {}
    for st in sorted(set(strata)):
        todo = sorted(u for u in size if stratum_of[u] == st)
        rng.shuffle(todo)
        total = sum(size[u] for u in todo)
        count = dict.fromkeys(KEYS, 0)
        for u in todo:
            k = max(KEYS, key=lambda k: fractions[KEYS.index(k)] * total - count[k])
            assign[u] = k
            count[k] += size[u]
    out = {k: [] for k in KEYS}
    for i, u in enumerate(units):
        out[assign[u]].append(i)
    return out
