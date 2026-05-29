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

```python
from deepSTRF.datasets.audio import CRCNSAC1Dataset

ds = CRCNSAC1Dataset(
    path="~/Documents/NRFdatasets/Audio/CRCNS_AC1",
    experimenter=("wehr", "asari"),   # default both; tuple or single str
    sites=("A1", "MGB"),              # default both
    signal_type="subthresh",          # alt: 'spikes' for Hann-smoothed PSTH proxy
    dt_ms=5.0,                        # matches Rançon 2025 Fig 2 settings
    download=False,                   # True needs $CRCNS_USERNAME / $CRCNS_PASSWORD
)
```

## Subset summary

|        | Recording        | sf       | Subjects   | Stimuli                           | N (loader) |
|--------|------------------|----------|------------|-----------------------------------|------------|
| Wehr   | whole-cell A1    | 4 kHz    | anaesth. rat | 15-s natural-sound fragments  | 25         |
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
``param.stimulus[i].file`` lookup table — typically class<N>/<base>.mat
plus suffixed amplitude / frequency variants). The loader resolves the
exact segment files and records them in ``stim_meta['segment_files']``.

## Processing pipeline

The loader is fully Python (no MATLAB runtime). Each instantiation:

1. **Auto-extracts** the three CRCNS-AC1 zips if not already done
   (``crcns-ac1.zip``, ``crcns-ac1-asari-results-{1,2}.zip``). Pass
   ``download=True`` (and CRCNS credentials) to fetch them from the
   NERSC mirror first.
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
- **Tone tuning curves and synthetic stimuli are not loaded in v1.**
  Only ``naturalsound`` (Wehr) / ``naturalsound`` sequences (Asari) are
  ingested. Adding tones is a future extension; gated by a
  ``stimuli=`` kwarg.
