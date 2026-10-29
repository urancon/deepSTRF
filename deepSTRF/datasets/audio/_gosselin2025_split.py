"""Leak-aware train / val / test split for Gosselin 2025.

Kept together (one unit): (1) sounds that are the same waveform up to a gain
(the 50 / 70 dB copies, from ``_gosselin2025_stim_groups.json``); (2) a
decomposed clip's reconstruction and all its components, since the
reconstruction is assembled from the components' audio. The original clip
only resembles its reconstruction (no shared audio), so it goes with its own
level copy. Everything else, including time-reversed clips and tones, sweeps
and ramps at other frequencies or directions, is its own unit. Units are drawn per stratum (the CSV
category, or ``decomposition``) to the set furthest below its target share.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from deepSTRF.datasets.audio._trial_audio import KEYS, draw_split  # noqa: F401 (re-exported)


def split_units(stim_meta: Sequence[dict], gain_copies: Sequence[Sequence[str]]
                ) -> Tuple[List[str], List[str]]:
    """Per stimulus: the unit it is drawn with, and that unit's stratum."""
    rep = {}
    for g in gain_copies:
        for n in g:
            rep[n] = sorted(g)[0]
    units, strata = [], []
    for m in stim_meta:
        if m["category"] == "decompositions":
            units.append(f"decomp:{m['source_clip']}")
            strata.append("decomposition")
        else:
            units.append(rep.get(m["name"], m["name"]))
            strata.append(m["category"])
    return units, strata
