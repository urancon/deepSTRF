# CRCNS AC1 Dataset (Wehr + Asari subsets)

**Dataset Source:** [AC1 Dataset](https://crcns.org/data-sets/ac/ac-1/about)

**Citation:**
```text
Asari, H; Wehr, M; Machens, C; Zador, A. (2009).
Auditory cortex and thalamic neuronal responses to various natural and synthetic sounds.
CRCNS.org. http://dx.doi.org/10.6080/K0KW5CXR
```

**Papers Using the Dataset:**

- Machens, C. K., Wehr, M. S., & Zador, A. M. (2004).
  ["Linearity of Cortical Receptive Fields Measured with Natural Sounds."](https://doi.org/10.1523/JNEUROSCI.4445-03.2004)
  *J. Neurosci.* 24(5):1089-1100. (the *Wehr* subset)
- Asari, H. & Zador, A. M. (2009).
  ["Long-Lasting Context Dependence Constrains Neural Encoding Models in Rodent Auditory Cortex."](https://doi.org/10.1152/jn.00577.2009)
  *J. Neurophysiol.* 102(5):2638-2656. (the *Asari* subset)
- Rançon, U., Masquelier, T., & Cottereau, B. R. (2025).
  ["Temporal recurrence as a general mechanism to explain neural responses in the auditory system."](https://doi.org/10.1038/s42003-025-08858-3)
  *Commun. Biol.* 8:1456.


## What you get

A single :class:`~deepSTRF.datasets.audio.CRCNSAC1Dataset` exposes both
historical subsets through the modern deepSTRF data paradigm
(``(B, N, R, T)`` outputs, NaN sentinels for sparse coverage). The
``experimenter`` argument filters between the two subsets; ``sites``
filters Asari's A1 vs MGB recordings (Wehr is all A1).

See the **[inspect_crcns_ac1 notebook](../ipynb/inspect_crcns_ac1.ipynb)**
for a guided tour: subthreshold Vm vs spike counts, trial-to-trial
variability, Wehr fragments vs Asari sequences, how the recording mode sets
the signal type, and the under-the-hood MedGauss detrend + artifact gating.

```python
from deepSTRF.datasets.audio import CRCNSAC1Dataset

ds = CRCNSAC1Dataset(
    path="~/Documents/NRFdatasets/Audio/CRCNS_AC1",
    experimenter=("wehr", "asari"),   # default both; tuple or single str
    sites=("A1", "MGB"),              # default both
    dt_ms=5.0,                        # matches Rançon 2025 Fig 2 settings
    download=False,                   # True needs $CRCNS_USERNAME / $CRCNS_PASSWORD
)
```

### Signal type is set by the recording mode (not a user choice)

The response signal is **determined by how each cell was recorded**, because
the recording mode dictates what signal physically exists:

| recording mode | subset | `nrn_meta['signal_type']` | sign | pair with |
|----------------|--------|---------------------------|------|-----------|
| whole-cell | Wehr, Asari A1 | `'subthresh'` (MedGauss-detrended Vm) | signed | `mse_loss` |
| cell-attached | Asari MGB | `'spikes'` (Hann-smoothed rate) | ≥ 0 | `poisson_loss` |

Cell-attached recordings have no intracellular Vm; whole-cell recordings have
no spikes (blocked in Wehr, not analysed in Asari A1 per the paper). The
loader derives the type per cell from `recording_type`, stores it in
`nrn_meta['signal_type']`, and exposes the cohort-level `ds.signal_type`
(`'subthresh'`, `'spikes'`, or `'mixed'`). Loading A1 + MGB together yields a
`'mixed'` cohort (signed-mV and spike-rate neurons side by side) and emits a
warning — filter by site / `signal_type` before training one model across
both.

## Subset summary

|        | Recording        | sf       | Subjects   | Stimuli                           | N (loader) |
|--------|------------------|----------|------------|-----------------------------------|------------|
| Wehr   | whole-cell A1    | 4 kHz    | anaesth. rat | 15-s fragments (2002 cells), 18–31-s streams (2003 cells) | 25 |
| Asari A1 | whole-cell A1  | 10 kHz   | anaesth. rat | spliced *sequences* of segments | 39         |
| Asari MGB | cell-attached MGB | 10 kHz | anaesth. rat | same as Asari A1 (subset)     | 14         |

(The "N (loader)" numbers reproduce the Asari 2009 paper counts —
39 A1 + 14 MGB cells tested with natural-sound ensembles, plus the 25
Wehr sessions used by Rançon 2024/2025.)

Wehr stimuli are documented by their fragment name (``humpback whale``,
``jaguar mating call``, etc.) and live under
``Stimuli/{fragments,category{1,2,3}}/<idx>.mat``. Asari stimuli are
*sequences* of 5 segments spliced together (e.g.
``'Sequence 1: 2  1  3  1  4'`` indexes into the recording's own
``param.stimulus[i].file`` lookup table — ``class<N>/<n>.mat`` plus short
``<n>s`` and long ``<n>l`` cuts of the same recording). The loader resolves
the exact segment files and records them in ``stim_meta['segment_files']``.

## Processing pipeline

The loader is fully Python (no MATLAB runtime). Each instantiation:

1. **Auto-extracts** the three CRCNS-AC1 zips if not already done
   (``crcns-ac1.zip``, ``crcns-ac1-asari-results-{1,2}.zip``). Pass
   ``download=True`` (and CRCNS credentials) to fetch them from the
   CRCNS download server first.
2. **Walks** ``Results/`` directories; one cell per session.
3. **Detrends** each Vm repeat with a MedGauss subtraction (median
   window 100 ms, gaussian σ 10 ms) — same family as the Rançon 2025
   supplement's "Detrending with MedGauss filter" recipe.
4. **Gates** repeats against three artifact tests
   (:class:`~deepSTRF.datasets.audio._crcns_ac1_native.RepeatGating`):
   amplitude clipping (``|Vm| > abs_mv_max``), sustained baseline
   steps (movement / dropout artifacts the MedGauss baseline can't
   track), and cross-trial Pearson disagreement. Bad repeats drop;
   the rejection counts are exposed at
   ``ds._rejection_counter`` for diagnostics.
5. **Spectrograms** each unique stimulus once via a Goertzel STFT at
   log-spaced frequencies, **directly at the target temporal resolution**
   (``hop = round(dt_ms × sf_stim / 1000)``) — no two-step
   compute-then-downsample as the legacy MATLAB pipeline did.

For the **cell-attached MGB** cohort, the subthreshold step is replaced by
spike extraction: drift-detrend → high-pass (subtract a **10 ms** median, the
Asari 2009 value) → threshold at 2.5 σ → 21 ms Hann smooth. On a sample MGB
cell this yields ~12 Hz evoked rate, matching the paper's 11.4 ± 16.9 Hz.

## Wehr: shared audio and `group_split`

The Wehr stimuli are 18 *fragment* files (15 s each, one source each; the
Hendrix file is 32 s) and 33 *streams* of 18–31 s. The streams are grouped
by the frequency range of their sounds (``category1`` 0–1.6 kHz,
``category2`` 0–6.4 kHz, ``category3`` 0–22 kHz). The labels are spectral
classes, not filters. A stream is a series of pieces from different sources,
cut on a 0.5-s grid with short fades at the joins.

Some files share audio:

- The first ~10 s of 13 fragment files reappear inside 14 streams; for
  example, the tungara frog fragment opens ``category2/18``.
- Fragments 12 and 13 are themselves built from fragment 8 and a screaming
  piha recording.
- Streams share no audio with each other.

The 10 cells recorded in 2002 heard only fragments, and the 15 cells
recorded in 2003 heard only streams. A model fitted to all cells therefore
sees the same audio in training and test whenever a fragment and a stream
that copies it fall in different sets. ``WEHR_SHARED_AUDIO`` lists the 9
groups of files that share audio, and ``stim_meta['overlap_group']`` gives
each Wehr stimulus its group (37 groups over the 51 files).

``CRCNSAC1Dataset.group_split(seed, fractions=(0.7, 0.1, 0.2))`` returns
train / val / test stimulus indices with every group in one set. The
stimuli stay whole, so their long context, and the adaptation it drives,
is kept. Groups holding a fragment file are drawn together (the streams
that copy them follow them); the other streams are drawn per category. At
``seed=0`` the split is 33 / 4 / 14 files, every category has test files,
24 of the 25 cells heard at least one test file, and 17 heard at least one
val file (3 of the 10 fragment cells: each of them heard only 3–11
fragments).

```python
from torch.utils.data import Subset
ds = CRCNSAC1Dataset(experimenter="wehr", sites="A1", dt_ms=5.0)
split = ds.group_split(seed=0)
train = Subset(ds, ds.stim_to_iter_indices(split["train"]))
```

## Asari: fragment-level stimuli and `group_split`

Asari & Zador (2009) built each stimulus set from N natural-sound fragments
and played sequences of them with no gap (5-ms ramps at the joins), so that
each fragment is heard after many different preceding fragments: the
design measures context dependence. A fragment therefore recurs in several
sequences of its set (about 3–4 times in the 7-fragment set heard by most
A1 cells). Any split of the *sequences* puts the same fragments in training
and test, so no leak-free split exists at the default
``asari_level='sequence'``, and ``group_split`` raises there.

``asari_level='segment'`` makes one stimulus per fragment file:

- every sequence is cut at its fragment boundaries (the cumulative file
  lengths, exactly as the loader splices it), and the trials of all the
  occurrences of a fragment are pooled as its repeats (A1: median 28
  trials per fragment and cell, MGB: 100);
- the context before a fragment is not part of the stimulus, so its effect
  on the response counts as trial-to-trial variability: a model cannot
  learn it, and the noise ceiling (``cc_norm``) treats it as noise;
- occurrences the recording stops inside are dropped (a few sequences end
  early).

``stim_meta`` gives each fragment its ``source``: the CD track it was cut
from (``'<CD>/<track>'``, parsed from the file's description), shared by the
base file and its short and long cuts. ``occurrences`` lists where it was
heard: ``(sequence, position, previous fragment, onset)``.

``group_split(seed)`` keeps every source track in one set, at A1 and MGB
alike (they share tracks). Tracks are drawn set by set, over the sets of
tracks the cells heard, largest first, so the 7-track set heard by most A1
cells gets val and test tracks. At ``seed=0``:

| Site | fragments train / val / test | cells with test data | cells with val data |
|------|------------------------------|----------------------|---------------------|
| A1   | 30 / 5 / 10                  | 38 / 39              | 35 / 39             |
| MGB  | 16 / 3 / 7                   | 14 / 14              | 8 / 14              |

```python
ds = CRCNSAC1Dataset(experimenter="asari", sites="A1", dt_ms=5.0,
                     asari_level="segment")
split = ds.group_split(seed=0)
```

## Reproducibility constants

For Rançon 2024/2025 numbers, the loader exports two constants:

- ``WEHR_VALID_NEURONS`` — the 21 of 25 cells used in those papers
  (drops the unresponsive ``1, 2, 4, 8`` per Machens et al. 2004).
- ``WEHR_NEURONS_SPLIT_NATURAL`` — per-neuron ``(train, val, test)``
  stim counts used in those papers (indexed by ``_wehr_cell_idx`` in
  ``nrn_meta``).

```python
from deepSTRF.datasets.audio import (
    CRCNSAC1Dataset, WEHR_VALID_NEURONS, WEHR_NEURONS_SPLIT_NATURAL,
)
ds = CRCNSAC1Dataset(experimenter="wehr", dt_ms=5.0)
ds.select_pop_by_nrn_predicate(
    lambda n: n["_wehr_cell_idx"] in WEHR_VALID_NEURONS
)
```

Per-cell single-unit fits over this cohort reproduce the published Wehr
numbers: **StateNet-GRU ≈ 0.31** and **Linear STRF ≈ 0.21** mean test
``cc_norm`` (Rançon et al. 2025, Table 1). For the Linear model use
``temporal_window_size ≈ 20–40`` at ``dt_ms=5.0`` (a 100–200 ms window) —
a stateless STRF needs the long window to reach that score, whereas the
GRU gets its temporal context from recurrence (``T=1`` input).

## Spectrogram defaults

Both subsets share the same Goertzel STFT layout: 53 log-spaced bands
from 100 Hz to 45 kHz at 6 bins/octave (the Asari 2025 setting). Wehr
stimuli (content ≤ 22 kHz) get the same F; the top ~5 bands sit near
the noise floor for them and are harmless. To recover the Wehr 2024
49-band setting exactly, pass ``fmax=25600.0`` to the constructor.

## Caveats

- **CRCNS account required for `download=True`** (free registration at
  https://crcns.org/register; credentials can be provided via
  ``$CRCNS_USERNAME`` / ``$CRCNS_PASSWORD``).
- **Vm drift + motion artifacts** are common in these recordings;
  several repeats per stimulus get gated out by the artifact tests.
  Inspect ``ds._rejection_counter`` to see the breakdown.
- **Numerical fidelity vs Rançon 2024 / 2025.** The Goertzel
  spectrogram is bit-equivalent (~fp64) to MATLAB's ``logspectrogram``,
  but the response cleanup pipeline differs in detail (we run
  artifact gating that the MATLAB pipeline did not). Expect model
  ``cc_norm`` within seed-level noise of the published numbers.
- **MGB spike window vs the prior pipeline.** The legacy ``asari.py``
  high-passed with ``median_filter(resp, 10)`` — 10 *samples* = 1 ms at
  10 kHz, a units bug (its own comment cited "10 ms (cf. Asari et al.)")
  that erased the ~1 ms spikes and left the threshold chasing noise. We
  use the paper-intended **10 ms**, which recovers the published MGB
  firing rate. So MGB spike-rate numbers here will differ from a
  pipeline that reproduced the 1 ms bug.
- **Tone tuning curves and synthetic stimuli are not loaded in v1.**
  Only ``naturalsound`` (Wehr) / ``naturalsound`` sequences (Asari) are
  ingested. Adding tones is a future extension; gated by a
  ``stimuli=`` kwarg.
