import os
import re
from typing import Optional

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._nat4_native import (
    epoch_names_matching,
    extract_epoch,
    load_per_site_recording,
    load_pop_recording,
    normalize_log1p_minmax_inplace,
    normalize_minmax_inplace,
)
from deepSTRF.utils.data_download import (
    default_cache_dir,
    unzip,
    zenodo_download,
)


# NAT4 Zenodo record (https://doi.org/10.5281/zenodo.8044773), public.
NAT4_ZENODO_RECORD = 8044773


# NEMS cell ids in NAT4 follow the convention <site>-<electrode>-<unit>, e.g.
# 'ARM029a-01-1'. The site itself is <3-letter animal code><digits><session>,
# e.g. 'ARM029a' (animal 'ARM', recording 029, session 'a').
_CELL_ID_RE = re.compile(r"^([A-Za-z]{3})\d+[a-z]?-(\d+)-(\d+)$")


def _parse_nat4_cell_id(cell_id: str) -> dict:
    """Best-effort decomposition of a NAT4 cell id.

    Returns a dict with ``site``, ``animal``, ``electrode``, ``unit_in_electrode``.
    Any field whose source is missing or unparseable is set to ``None``.
    """
    out = {"site": None, "animal": None, "electrode": None, "unit_in_electrode": None}
    if not isinstance(cell_id, str) or "-" not in cell_id:
        return out
    parts = cell_id.split("-")
    out["site"] = parts[0]
    m = _CELL_ID_RE.match(cell_id)
    if m is None:
        # site looked sensible (first segment), but electrode/unit didn't parse.
        return out
    out["animal"] = m.group(1)
    out["electrode"] = int(m.group(2))
    out["unit_in_electrode"] = int(m.group(3))
    return out


def download_nat4(area: str, dest: Optional[str] = None) -> str:
    """Download the NAT4 release from Zenodo into ``dest``.

    Fetches the population .tgz, the per-cell auditory CSV, and the per-site
    .zip. The single-sites zip is unpacked into ``<dest>/<area>_single_sites/``
    so the loader finds the per-site .tgzs where it expects them.

    Idempotent: skips files / dirs that already exist.

    Parameters
    ----------
    area : {'A1', 'PEG'}
    dest : str, optional
        Defaults to ``default_cache_dir('NAT4')`` (overridable via
        ``$DEEPSTRF_DATA_DIR``).
    """
    assert area in ("A1", "PEG"), f"area must be 'A1' or 'PEG' (got {area!r})"
    dest_path = str(default_cache_dir("NAT4") if dest is None else dest)
    os.makedirs(dest_path, exist_ok=True)

    pop_tgz_name = f"{area}_NAT4_ozgf.fs100.ch18.tgz"
    pop_tgz_path = os.path.join(dest_path, pop_tgz_name)
    if not os.path.exists(pop_tgz_path):
        zenodo_download(NAT4_ZENODO_RECORD, pop_tgz_name, pop_tgz_path)

    csv_name = f"{area}_pred_correlation.csv"
    csv_path = os.path.join(dest_path, csv_name)
    if not os.path.exists(csv_path):
        zenodo_download(NAT4_ZENODO_RECORD, csv_name, csv_path)

    single_sites_dir = os.path.join(dest_path, f"{area}_single_sites")
    if not os.path.isdir(single_sites_dir):
        zip_name = f"{area}_single_sites.zip"
        zip_path = os.path.join(dest_path, zip_name)
        if not os.path.exists(zip_path):
            zenodo_download(NAT4_ZENODO_RECORD, zip_name, zip_path)
        unzip(zip_path, dest_path)

    return dest_path


class NAT4_Dataset(AudioNeuralDataset):
    """A PyTorch dataset for NAT4 (Pennington & David, 2022 / 2023).


    =============== SOURCE ================

    See original papers for details:
     - "Can deep learning provide a generalizable model for dynamic sound
       encoding in auditory cortex?" Pennington & David. (2022 preprint)
     - "A convolutional neural network provides a generalizable model of
       natural sound coding by neural populations in auditory cortex"
       Pennington & David, PLOS Computational Biology (2023).

    Data freely available at https://doi.org/10.5281/zenodo.8044773 (no
    account required) — auto-fetched by ``NAT4_Dataset(download=True)``.


    =============== DETAILS ================

    - Two cortical areas: ``A1`` (primary, 849 cells of which 777 auditory)
      and ``PEG`` (secondary, 398 of which 339 auditory). Pass ``area=...``;
      one instance covers one area. To pool both, instantiate twice and
      ``concat_neural_datasets([a1, peg])``.
    - 595 stimuli total: 18 high-rep (``val``, 20 trials) + 577 low-rep
      (``est``, 1 trial). Each clip is 1.5 s.
    - Time bin: ``dt_ms = 10`` (the population recording is precomputed at
      fs=100 with val pre-averaged over 20 reps; per-site spike trains are
      at fs=1000 and downsampled to 10 ms by summing).
    - Spectrogram: F = 18 ozgf bands, T = 150 frames per stim.

    The loader reads the published NAT4 archive directly with native CSV
    / JSON / HDF5 parsers — no NEMS0 dependency. The 4 NEMS calls used by
    the legacy loader (``load_recording``, ``epoch_names_matching``,
    ``xforms.normalize_sig``, ``preprocessing.split_pop_rec_by_mask``)
    are reimplemented in ``deepSTRF.datasets.audio._nat4_native``.


    =============== STRUCTURE ================

    Follows the standard deepSTRF data paradigm (see docs/_source/md/data_paradigm.md).
    NAT4-specific metadata contents:
     - self.stims                       list of S=595 tensors (1, F=18, T=150)
     - self.responses                   list of S lists of N tensors —
                                        est stims have shape (R=1, T=150),
                                        val stims have shape (R=20, T=150);
                                        ``(1, 1)`` NaN sentinel for the (s, n)
                                        pairs where the cell wasn't recorded
                                        for that stim (cells from sites that
                                        only saw a subset of the stim bank).
     - self.stim_meta                   list of S dicts {"name", "subset"}
                                        where subset is 'est' or 'val'. The
                                        ``subset='all'|'est'|'val'`` constructor
                                        arg filters this list at load time.
     - self.neuron_metadata             list of N dicts with the per-cell info
                                        deepSTRF can recover from the archive:
                                        ``cell_id`` (raw NEMS id, e.g.
                                        ``'ARM029a-01-1'``), ``area``,
                                        ``auditory`` (flag from the dataset's
                                        ``<area>_pred_correlation.csv``),
                                        plus the parsed components ``site``
                                        (e.g. ``'ARM029a'``), ``animal`` (3-char
                                        prefix of the site, e.g. ``'ARM'``),
                                        ``electrode`` (int when the id parses
                                        cleanly), and ``unit_in_electrode``
                                        (int). Components default to ``None``
                                        for any cell whose id does not match
                                        the standard ``<site>-<elec>-<unit>``
                                        scheme.

    """

    def __init__(self, path: Optional[str] = None, area: str = 'A1',
                 dt_ms: float = 10.0, smooth: bool = False,
                 download: bool = False, subset: str = 'all'):
        """
        Parameters
        ----------
        path : str, optional
            Path to the NAT4 data folder. Defaults to the platformdirs cache.
        area : {'A1', 'PEG'}
            Cortical area.
        dt_ms : float, default 10.0
            Time-bin width in ms. Currently must equal 10.0; the population
            recording is precomputed at fs=100 and the per-site downsampling
            assumes a fixed 10x ratio from fs=1000.
        smooth : bool, default False
            If True, smooth PSTHs with a 21 ms Hanning window. Off by default
            here because NAT4 trials are typically used as-is for STRF
            fitting (unlike CRCNS-AA where smoothing is the published norm).
        download : bool, default False
            If True and the data is missing under ``path``, fetch it from
            Zenodo (record 8044773).
        subset : {'all', 'est', 'val'}, default 'all'
            If 'est' or 'val', only that stimulus subset is loaded —
            ``stim_meta`` / ``stims`` / ``responses`` shrink accordingly,
            and the (more expensive) per-site spike-time pass is skipped
            entirely under ``subset='est'``. The two subsets correspond to
            Pennington & David's published estimation set (575 stims, R=1,
            from the population recording) and validation set (18 stims,
            R=20, from the per-site recordings) respectively. Note that 33
            of the 849 A1 cells have no val data — under ``subset='val'``
            their responses are full NaN sentinels; pair the constructor
            arg with ``ds.select_pop_by_stim_attr('subset', 'val')`` to
            drop them automatically (idiomatic alternative:
            ``ds.select_stims_by_attr('subset', 'val')`` — which leaves the
            full stim bank loaded but applies the bidirectional rule, so
            cells without val data are hidden from ``__getitem__``).
        """

        assert area in ("A1", "PEG"), \
            f"Unexpected area {area!r}, choose between 'A1' or 'PEG'"
        assert subset in ("all", "est", "val"), \
            f"Unexpected subset {subset!r}, choose between 'all', 'est', 'val'"
        assert dt_ms == 10.0, (
            f"NAT4 spectrograms are precomputed at dt=10 ms; got dt_ms={dt_ms}. "
            f"Re-rasterizing the responses is straightforward but the "
            f"spectrogram .tgz would also need re-binning (TODO)."
        )

        if path is None:
            path = str(default_cache_dir("NAT4"))
        if download:
            download_nat4(area, path)

        super().__init__(path, dt_ms)
        self.area = area
        self.species = 'ferret'
        self.F = 18

        # =========  LOAD THE POPULATION RECORDING (used for est, R=1)  ===========

        # Accept either the .tgz archive OR an already-extracted directory.
        tgz_path = os.path.join(path, f'{area}_NAT4_ozgf.fs100.ch18.tgz')
        dir_path = os.path.join(path, f'{area}_NAT4_ozgf.fs100.ch18')
        if os.path.exists(tgz_path):
            datafile = tgz_path
        elif os.path.isdir(dir_path):
            datafile = dir_path
        else:
            raise FileNotFoundError(
                f"NAT4 expects either {tgz_path} or {dir_path}/. "
                f"Pass download=True to fetch the .tgz from Zenodo, or "
                f"place the data manually."
            )
        rec = load_pop_recording(datafile)

        # log1p + minmax for the spectrogram, plain minmax for the response —
        # matches the preprocessing baked into the published Pennington &
        # David models. Both operations are global (single (min, max) per
        # signal across all of T × K).
        normalize_log1p_minmax_inplace(rec)
        normalize_minmax_inplace(rec)

        cells = rec.chans
        val_sounds = epoch_names_matching(rec.epochs, "^STIM_00cat")
        est_sounds = epoch_names_matching(rec.epochs, "^STIM_cat")

        # =========  STIM SPECTROGRAMS (est first, then val)  ===========

        load_est = subset in ("all", "est")
        load_val = subset in ("all", "val")

        self.stim_meta = []
        self.stims = []
        if load_est:
            for est_sound in est_sounds:
                spec = extract_epoch(rec, 'stim', est_sound)  # (R=1, F, T)
                self.stims.append(torch.from_numpy(spec[0]).unsqueeze(0).float())  # (1, F, T)
                self.stim_meta.append({'name': est_sound, 'subset': 'est'})
        if load_val:
            for val_sound in val_sounds:
                spec = extract_epoch(rec, 'stim', val_sound)
                self.stims.append(torch.from_numpy(spec[0]).unsqueeze(0).float())
                self.stim_meta.append({'name': val_sound, 'subset': 'val'})

        # =========  NEURON METADATA (auditory flag + parsed cell_id)  ===========

        self.neuron_metadata = []
        list_neurons = pd.read_csv(os.path.join(path, f'{area}_pred_correlation.csv'))
        cell_to_aud = dict(zip(list_neurons['cellid'], list_neurons['sig_auditory']))
        for cell in cells:
            self.neuron_metadata.append({
                'cell_id': cell,
                'area': area,
                'auditory': bool(cell_to_aud.get(cell, False)),
                **_parse_nat4_cell_id(cell),
            })
        self.N_neurons = len(self.neuron_metadata)

        # =========  EST RESPONSES (1 trial per stim, full population)  ===========
        # Cells that didn't see a given est stim get a (1, 1) NaN sentinel
        # rather than a (1, T) trace of NaNs / zeros.

        est_responses_per_stim = []  # list of S_est lists of N (1, T) tensors / NaN
        if load_est:
            for est_sound in est_sounds:
                arr = extract_epoch(rec, 'resp', est_sound)  # (R=1, N, T)
                stim_resps = []
                for n in range(self.N_neurons):
                    trace = arr[:, n, :]  # (1, T)
                    if np.isnan(trace).all():
                        stim_resps.append(torch.full((1, 1), float('nan')))
                    else:
                        stim_resps.append(torch.from_numpy(trace).float())
                est_responses_per_stim.append(stim_resps)

        del rec

        # =========  VAL RESPONSES (20 trials per stim, per-site stitching)  ===========
        # The pop rec averages val over 20 reps and only keeps R=1; for trial-
        # resolved data we go to the per-site .tgzs at fs=1000 and downsample.

        val_responses_per_stim = []
        if load_val:
            val_files = sorted(os.listdir(os.path.join(path, f'{area}_single_sites')))

            per_site_val = []          # list of (S_val, R, N_subpop, T) tensors
            val_cells_in_order = []    # cell-id order across stitched per-site recs
            for filename in tqdm(val_files, desc=f'NAT4 {area} val sites'):
                site_path = os.path.join(path, f'{area}_single_sites', filename)
                site_rec = load_per_site_recording(site_path)
                val_cells_in_order += list(site_rec.chans)

                site_responses = []
                for val_sound in val_sounds:
                    # extract at fs=1000 -> (R=20, N_subpop, T_ms=1500)
                    arr = extract_epoch(site_rec, 'resp', val_sound)
                    R, N_subpop, T_ms = arr.shape
                    # downsample 1 ms -> 10 ms by summing
                    arr = arr.reshape(R, N_subpop, -1, 10).sum(axis=-1)
                    site_responses.append(torch.from_numpy(arr).float())
                per_site_val.append(torch.stack(site_responses))  # (S_val, R, N_subpop, T)
                del site_rec

            val_full = torch.cat(per_site_val, dim=2)         # (S_val, R, N_total_in_val_order, T)
            val_full = val_full.permute(0, 2, 1, 3)           # (S_val, N, R, T)

            # Cells in the per-site stitching order are not in the same order as
            # the population's `cells` list (and the val concatenation may include
            # cells the pop rec doesn't have, or vice versa). Reindex onto `cells`;
            # any pop-rec cell missing from the val concat gets a NaN sentinel.
            index_map = {u: i for i, u in enumerate(val_cells_in_order)}
            S_val = val_full.shape[0]
            for s in range(S_val):
                stim_resps = []
                for n, cell in enumerate(cells):
                    if cell in index_map:
                        stim_resps.append(val_full[s, index_map[cell]])
                    else:
                        stim_resps.append(torch.full((1, 1), float('nan')))
                val_responses_per_stim.append(stim_resps)

        # est first, then val — matches the order of self.stims / self.stim_meta
        self.responses = est_responses_per_stim + val_responses_per_stim

        if smooth:
            self.smooth_responses(window_ms=21.0)

        self.validate()
