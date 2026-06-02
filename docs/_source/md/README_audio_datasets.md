## Auditory datasets

Currently, **11 response datasets** are available, each with a short name for
easier reference. They span five species and a range of recording modalities —
extracellular single-units, multi-unit activity, and scalp EEG — but all share
the same `NeuralDataset` API (NaN-sentinel missing trials, optional
`download=True`, and the selection / concatenation utilities). Each dataset has
its own page below with the original paper, licensing, response shape, and
instructions on how to obtain the data (auto-download where possible).

### Ferret (*Mustela putorius furo*)
* [NS1](README_NS1.md) — primary auditory cortex (A1); 20 clips of natural sounds (speech, ferret & other animal vocalizations, environmental).
* [NAT4](README_NAT4.md) — A1 & PEG (849 + 398 units); natural sounds.
* [Espejo 2019](README_Espejo.md) — awake A1; natural sounds + vocalization-modulated noise (two disjoint releases).
* [Wingert 2026](README_Wingert2026.md) — A1 + PEG (plus smaller AC & HC fields) single-units across 4 ferrets.

### Zebra finch (*Taeniopygia guttata*)
* [CRCNS AA1](README_CRCNS_AA1.md) — conspecific vocalizations + flat ripples.
* [CRCNS AA2](README_CRCNS_AA2.md) — extracellular single-units (57 birds); conspecific vocalizations, flat & song ripples.
* [CRCNS AA4](README_CRCNS_AA4.md) — extracellular recordings spanning the full zebra-finch vocal repertoire (songs + calls) plus ripple noise.
* [Le 2025](README_Le_2025.md) — auditory cortex; occluded conspecific song (auditory-restoration paradigm).

### Rat (*Rattus norvegicus*)
* [CRCNS AC1 (Wehr + Asari)](README_CRCNS_AC1.md) — auditory cortex (A1) & thalamus (MGB); natural & synthetic sounds.

### Squirrel monkey (*Saimiri sciureus*)
* [Downer 2025](README_Downer2025.md) — A1 multi-unit activity; TIMIT speech + species-specific vocalizations.

### Human (*Homo sapiens*)
* [Alice EEG](README_Alice_EEG.md) — scalp EEG; naturalistic continuous speech (audiobook; Brodbeck 2023).


```{toctree}
---
hidden:
maxdepth: 1
caption: Datasets
---

README_NS1.md
README_NAT4.md
README_Espejo.md
README_Wingert2026.md
README_CRCNS_AA1.md
README_CRCNS_AA2.md
README_CRCNS_AA4.md
README_Le_2025.md
README_CRCNS_AC1.md
README_Downer2025.md
README_Alice_EEG.md
```
