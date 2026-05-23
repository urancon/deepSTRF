# Ferret AC encoding subspace (Wingert 2026)

**Dataset Source:** [Wingert et al. 2026 — Zenodo 18331549](https://doi.org/10.5281/zenodo.18331549)

**Original paper:**
- ["Convolutional neural network models describe the encoding subspace of local circuits in auditory cortex"](https://doi.org/10.1038/s41593-026-02216-0)
  by Jereme C. Wingert, Satyabrata Parida, Sam V. Norman-Haignere &
  Stephen V. David. *Nature Neuroscience* (2026).

## Dataset Details

**Description of Stimuli:**
- Concatenated natural-sound sequences drawn from the Audioset Core 3 Complete
  corpus and the Pro Sound Effects library, crossfaded with a 10 ms Hanning
  window. Each sequence is **20 s of sound** preceded / followed by **1 s of
  silence** at the older recording cohort (47 sites = 20 s window, no silence
  flanks) or recorded as **22 s = 1 s pre + 20 s sound + 1 s post** at the
  newer cohort (21 sites). Both cohorts share the same gtgram bins (`dt = 10
  ms`, `F = 32` log-spaced bands from 200 Hz to 20 kHz, log-compressed).
- Each site presents ~100 unique single-rep estimation sequences (one
  presentation each, `STIM_seqNNNN.wav`) plus a subset of up to 6
  test sequences (`STIM_00seq*.wav`) with **R varying 5–30 across sites**.
  The 6 test files share their source audio but each session re-rasterizes
  its own copy — loader treats `(session, stim_name)` as the canonical
  stim key and emits separate stim entries per session, naturally
  handling both duration cohorts under the deepSTRF ragged-T paradigm.

**Description of Neurons:**
- 2 128 A1 + 746 PEG + 217 AC + 37 HC single-units across 67 recording
  sites in 4 ferrets. The paper headlines A1 + PEG; the smaller AC and HC
  subsets are exposed for completeness but documented as less-curated.
- 131 additional cells from three otherwise-unrepresented PRN sessions
  (PRN010b, PRN011b, PRN020b) ship without an area label. Use
  `include_unlabeled=True` to surface them.
- Sites range from 5 to 256 stimuli; cell counts per site span 8 to 65.
- Each cell carries the published `goodpred` flag (auditory-responsive
  to the sound set, ~79% of cells). The two-probe SLJ032a recording
  contributes 76 cells under `site='SLJ032a'` (probe A) plus 47 under
  `site='SLJ032a-B'` (probe B).

**Available Data:**
- One self-contained `.tgz` archive per recording site under
  `recordings/`, NEMS recording format (`<site>.meta.json`,
  `<site>.resp.h5`, `<site>.resp.epoch.csv`, `<site>.stim.h5`,
  `<site>.stim.epoch.csv`, `<site>.stim.json`). Spike trains stored as
  per-cell spike-time arrays (PointProcess) at fs = 100; spectrograms
  as a TiledSignal h5 with one `(F=32, T)` array per unique stim.
- `cell_list.csv` — per-cell metadata: `cellid`, `siteid`, `area`,
  `layer`, `depth`, `narrow`, `celltype`, `sw` (spike width), and
  `goodpred`. Authoritative source for the `site` field of `nrn_meta`.
- `wav.zip` (raw 44.1 kHz waveforms) and `models.zip` (published CNN /
  LN / subspace fits) are **not** used by deepSTRF and **not** fetched by
  `download=True`.

**deepSTRF parses the archive directly — `nems0` is not required.**


## Setup

Easiest path — auto-download from Zenodo into the platformdirs cache:

```python
from deepSTRF.datasets.audio import Wingert2026Dataset

ds_a1  = Wingert2026Dataset(area='A1',  download=True)   # ~4.35 GB recordings.zip
ds_peg = Wingert2026Dataset(area='PEG', download=True)   # (no extra download)
```

`download=True` is idempotent and fetches **only** the files the loader
needs (`recordings.zip` + `cell_list.csv`); it skips the 3.7 GB
`wav.zip` and the 0.1 MB `models.zip`. The default cache directory is
`platformdirs.user_cache_dir('deepSTRF')/Wingert2026`, overridable via
`$DEEPSTRF_DATA_DIR`. To use a custom path explicitly:

```python
ds = Wingert2026Dataset('/path/to/wingert2026/', area='A1', download=True)
```

If you already have the data laid out manually, just pass the path:

```python
ds = Wingert2026Dataset('/path/to/wingert2026/', area='A1')
```

Expected files in the data dir:
* `recordings/<SITE>_<hash>.tgz` (one per recording site)
* `cell_list.csv`


## Filters: area, site, subset

`area` and `site` compose by intersection. Both accept a string or an
iterable of strings; either can be `None` to pass through.

```python
# all 2128 A1 cells in 50 sessions, the headline cohort
ds = Wingert2026Dataset(area='A1')

# A1 + PEG together (the paper's "auditory cortex" cohort, 2874 cells)
ds = Wingert2026Dataset(area=['A1', 'PEG'])

# a single recording site
ds = Wingert2026Dataset(site='CLT027c')

# the two-probe SLJ032a session — both probes share the same .tgz, so
# loading both is one .tgz read (no stim duplication across probes)
ds = Wingert2026Dataset(site=['SLJ032a', 'SLJ032a-B'])

# opt-in 131 area=None cells from PRN010b / PRN011b / PRN020b
ds = Wingert2026Dataset(area=None, include_unlabeled=True)   # N = 3259
```

`subset='est'` keeps the single-rep `STIM_seq*` estimation stims;
`subset='val'` keeps the high-rep `STIM_00*` test stims.

**Per-site R for the test stims varies dramatically** (5–30 across the
release), and many sites only ever heard 1–2 of the 6 test stims. The
deepSTRF data paradigm handles this naturally — cells whose session
didn't present a given test stim get a `(1, 1)` NaN sentinel for that
`(stim, cell)` pair, and the bidirectional `select_stims_by_attr` rule
hides cells with no real data for the selected stim subset.

```python
ds = Wingert2026Dataset(area='A1')
ds.select_stims_by_attr('subset', 'val')   # cells with no val data hidden
```


## Per-cell metadata

`nrn_meta[n]` carries the cell-list-canonical fields plus parsed
cell-id components:

| Field                | Example          | Notes                                                          |
|----------------------|------------------|----------------------------------------------------------------|
| `cell_id`            | `'CLT027c-009-1'` | Raw cell id. 3-segment for most cells, 4-segment for SLJ032a. |
| `site`               | `'CLT027c'`      | Authoritative siteid from `cell_list.csv`. The two-probe SLJ032a recording assigns `'SLJ032a'` to probe A and `'SLJ032a-B'` to probe B. |
| `session`            | `'CLT027c'`      | Recording-session label (first dash-separated segment of `cell_id`). Equals `site` everywhere except SLJ032a-B, whose session is `'SLJ032a'`. |
| `area`               | `'A1'`           | `'A1'`, `'PEG'`, `'AC'`, `'HC'`, or `None` (only when `include_unlabeled=True`). |
| `layer`              | `'56'`           | Cortical layer string (`'1-3'`, `'4'`, `'56'`). `None` for unlabeled cells. |
| `depth`              | `500.0`          | Depth in μm relative to L3/4 boundary. `None` for unlabeled cells. |
| `narrow`             | `False`          | Putative-inhibitory flag (spike width < 0.35 / 0.375 ms cutoff, probe-dependent). `None` for unlabeled cells. |
| `celltype`           | `'RD'`           | One of `'RD'`, `'RS'`, `'ND'`, `'NS'` (Regular / Narrow × Deep / Superficial). `None` for unlabeled cells. |
| `sw`                 | `0.756`          | Spike width in ms. `None` for unlabeled cells. |
| `goodpred`           | `True`           | Published auditory-responsive flag (~79% of cells). Always populated. |
| `animal`             | `'CLT'`          | 3-letter animal code (`CLT`, `LMD`, `PRN`, `SLJ`). Parsed from cell id. |
| `electrode`          | `9`              | Probe-channel index. Parsed from cell id. |
| `unit_in_electrode`  | `1`              | Unit-on-channel index. Parsed from cell id. |

Use the standard filter API:

```python
ds.select_pop_by_nrn_attr('area', 'A1')                 # one area
ds.select_pop_by_nrn_attr('goodpred', True)             # auditory-responsive cells
ds.select_pop_by_nrn_attr('narrow', True)               # putative inhibitory
ds.select_pop_by_nrn_predicate(lambda n: n['depth'] is not None
                                          and n['depth'] < 200)  # superficial
```


## Per-stim metadata

`stim_meta[s]` carries:

| Field      | Example                | Notes                                                    |
|------------|------------------------|----------------------------------------------------------|
| `name`     | `'STIM_seq0032.wav'`   | Source-wav file name as it appears in the NEMS archive. |
| `subset`   | `'est'` / `'val'`      | File-name prefix: `STIM_00*` → `'val'`, else `'est'`.   |
| `session`  | `'CLT027c'`            | Recording session that presented this stim copy.        |

Stim tensors have shape `(1, F=32, T)` with `T ∈ {2000, 2200}` —
ragged on T by design (the two recording cohorts use different silence
flanks). The default `neural_collate` zero-pads on the right.


## Note on the PRN018a / PRN018b duplicate

The release ships **two .tgz files for site PRN018a** —
`PRN018a_*.tgz` and `PRN018b_*.tgz` — with bit-identical contents (same
40 cells, same 256 stims, same spike-time arrays). The loader detects
this on its first session-map scan and drops `PRN018b_*.tgz` with a
`UserWarning`. Set `filterwarnings("ignore", category=UserWarning)` to
silence it if needed.

Three other PRN sessions (`PRN015a`, `PRN017a`, `PRN018a`) have a
`.tgz` filename that doesn't match the cell-id prefix
(`PRN015b_*.tgz`, etc. — the file basename uses the next session
suffix). The loader resolves these by mapping on the cell-id prefix,
not the filename, so users see `'PRN015a'` everywhere in `nrn_meta`
and `stim_meta`.


## Memory and load-time expectations

The loader rasterizes spike trains into per-`(stim, cell)` `(R, T)`
tensors at construction time. Approximate cost on the local mirror
(Linux + Python 3.10 + 10 ms bins):

| Scope             | N     | S    | Load time | Peak RSS |
|-------------------|-------|------|-----------|----------|
| One small site    | 20    | 11   | ~7 s      | <0.5 GB  |
| Two adjacent sites | 77    | 117  | ~8 s      | ~0.8 GB  |
| 5 A1 sites        | 231   | 685  | ~15 s     | ~1.3 GB  |
| Full A1 (50 sites) | 2 128 | ~5 300 | ~3 min    | ~7–10 GB |

For repeated experimentation against the full cohort, persist the
dataset object once with `torch.save` and reload — instantiating a
fresh `Wingert2026Dataset(area='A1')` re-runs the rasterizer every
time. Or use `select_pop_by_nrn_attr` / `select_pop_by_nrn_predicate`
to filter down to the experimentally-relevant subset before training.

All `(1, 1)` NaN sentinels in the response grid share a **single
underlying tensor object** — the response grid for an A1 instance
costs ~80 MB of Python list pointers rather than the 5+ GB a naive
per-slot `torch.full(...)` would consume. The behaviour is enforced by
a regression test (`tests/test_wingert2026.py::test_sentinels_share_one_reference`).
