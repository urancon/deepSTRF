import gc
import os
from typing import Optional

from tqdm import tqdm

import torch
import pandas as pd

try:
    from nems0.recording import load_recording
    from nems0 import xforms, preprocessing, epoch
    _NEMS_AVAILABLE = True
except ImportError:
    _NEMS_AVAILABLE = False

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.utils.data_download import (
    default_cache_dir,
    unzip,
    zenodo_download,
)


_NEMS_INSTALL_HINT = (
    "NAT4_Dataset currently relies on the NEMS0 library to read its .tgz\n"
    "recording format. Install it with:\n"
    "    pip install 'deepSTRF[nems]'\n"
    "See https://github.com/LBHB/NEMS0 for details. A native (NEMS-free)\n"
    "loader is on the roadmap (cf. IDEAS.md)."
)

# NAT4 Zenodo record (https://doi.org/10.5281/zenodo.8044773), public.
NAT4_ZENODO_RECORD = 8044773


def download_nat4(area: str, dest: Optional[str] = None) -> str:
    """Download the NAT4 release from Zenodo into ``dest``.

    Fetches:
     - ``<area>_NAT4_ozgf.fs100.ch18.tgz``     (population recording, ~30 MB for A1)
     - ``<area>_pred_correlation.csv``         (per-cell auditory-responsive flag)
     - ``<area>_single_sites.zip``             (per-site .tgz files, ~70 MB for A1)

    The single-sites zip is unpacked into ``<dest>/<area>_single_sites/`` so
    the loader finds the per-site tgz files where it expects them.

    Idempotent: skips files / dirs that already exist.

    Parameters
    ----------
    area : {'A1', 'PEG'}
        Cortical area.
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
    - Time bin: ``dt_ms = 10`` (responses are stored at fs=100 in NEMS;
      val responses are downsampled from 1 ms by summing over 10 ms).
    - Spectrogram: F = 18 ozgf bands, T = 150 frames per stim.

    NB: this loader currently parses the NEMS recording format via the
    optional ``nems0`` extra (``pip install 'deepSTRF[nems]'``). A native,
    NEMS-free re-implementation is on the roadmap (cf. IDEAS.md).


    =============== STRUCTURE ================

    Follows the standard deepSTRF data paradigm (see docs/_source/md/data_paradigm.md).
    NAT4-specific metadata contents:
     - self.stims                       list of S=595 tensors (1, F=18, T=150)
     - self.responses                   list of S lists of N tensors —
                                        est stims have shape (R=1, T=150),
                                        val stims have shape (R=20, T=150);
                                        ``(1, 1)`` NaN sentinel for the (s, n)
                                        pairs flagged as null in the est set
                                        (neuron not recorded for that stim).
     - self.stim_meta                   list of S dicts {"name", "subset"}
                                        where subset is 'est' or 'val'.
     - self.neuron_metadata             list of N dicts {"cell_id", "area",
                                        "auditory"} — ``auditory`` is the
                                        per-cell flag from the dataset's
                                        ``<area>_pred_correlation.csv``.

    """

    def __init__(self, path: Optional[str] = None, area: str = 'A1',
                 dt_ms: float = 10.0, smooth: bool = False,
                 download: bool = False):
        """
        Parameters
        ----------
        path : str, optional
            Path to the NAT4 data folder. Defaults to the platformdirs cache.
        area : {'A1', 'PEG'}
            Cortical area.
        dt_ms : float, default 10.0
            Time-bin width in ms. NEMS-side rasterization is at fs=100 (10 ms);
            val responses are summed from fs=1000 down to dt_ms by integer
            divisor (so any ``dt_ms`` that divides 10 ms evenly is fine, but
            non-default values are not currently exposed because the
            spectrogram is precomputed at 10 ms in the .tgz).
        smooth : bool, default False
            If True, smooth PSTHs with a 21 ms Hanning window. Off by default
            here because NAT4 trials are typically used as-is for STRF
            fitting (unlike CRCNS-AA where smoothing is the published norm).
        download : bool, default False
            If True and the data is missing under ``path``, fetch it from
            Zenodo (record 8044773). Installs the optional [nems] extra is
            still required to parse the .tgz once it lands.
        """

        if not _NEMS_AVAILABLE:
            raise ImportError(_NEMS_INSTALL_HINT)
        assert area in ("A1", "PEG"), \
            f"Unexpected area {area!r}, choose between 'A1' or 'PEG'"
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

        # =========  LOAD THE POPULATION RECORDING (est set)  ===========

        datafile = os.path.join(path, f'{area}_NAT4_ozgf.fs100.ch18.tgz')
        rec = load_recording(datafile)

        context = {'rec': rec}
        # log-compress + minmax normalize the spectrogram (matches the
        # preprocessing baked into the published Pennington & David models).
        context.update(xforms.normalize_sig(sig='stim', norm_method='minmax', log_compress=1, **context))
        context.update(xforms.normalize_sig(sig='resp', norm_method='minmax', **context))
        context.update(preprocessing.split_pop_rec_by_mask(**context))

        cells = context['rec']['resp'].chans
        val_sounds = epoch.epoch_names_matching(context['rec']['resp'].epochs, "^STIM_00cat")
        est_sounds = epoch.epoch_names_matching(context['rec']['resp'].epochs, "^STIM_cat")

        # =========  STIM SPECTROGRAMS (est first, then val)  ===========

        self.stim_meta = []
        self.stims = []

        for est_sound in est_sounds:
            spec = context['rec']['stim'].extract_epoch(est_sound)  # (1, F, T)
            self.stims.append(torch.from_numpy(spec))
            self.stim_meta.append({'name': est_sound, 'subset': 'est'})

        for val_sound in val_sounds:
            spec = context['rec']['stim'].extract_epoch(val_sound)
            self.stims.append(torch.from_numpy(spec))
            self.stim_meta.append({'name': val_sound, 'subset': 'val'})

        # =========  NEURON METADATA (auditory flag from CSV)  ===========

        self.neuron_metadata = []
        list_neurons = pd.read_csv(os.path.join(path, f'{area}_pred_correlation.csv'))
        cell_to_aud = dict(zip(list_neurons['cellid'], list_neurons['sig_auditory']))
        for cell in cells:
            self.neuron_metadata.append({
                'cell_id': cell,
                'area': area,
                'auditory': bool(cell_to_aud.get(cell, False)),
            })
        self.N_neurons = len(self.neuron_metadata)

        # =========  EST RESPONSES (1 trial per stim, full population)  ===========
        # Cells that did not see a given est stim get a (1, 1) NaN sentinel
        # rather than a (1, T) trace of zeros — paradigm-compliant.
        # NEMS' extract_epoch returns (R, N, T) with NaN-for-missing for the
        # cross-site cells (they share the same time grid via stitching).

        est_responses_per_stim = []  # list of S_est lists of N (R=1, T) tensors / NaN
        for est_sound in est_sounds:
            arr = context['rec']['resp'].extract_epoch(est_sound)  # (1, N, T) numpy
            stim_resps = []
            for n in range(self.N_neurons):
                trace = arr[:, n, :]   # (1, T)
                # NEMS marks unrecorded (cell, stim) pairs with NaN; collapse
                # those whole-NaN traces to the canonical (1, 1) sentinel so
                # downstream code can rely on the deepSTRF paradigm.
                if torch.from_numpy(trace).isnan().all():
                    stim_resps.append(torch.full((1, 1), float('nan')))
                else:
                    stim_resps.append(torch.from_numpy(trace))
            est_responses_per_stim.append(stim_resps)

        del rec
        del context
        gc.collect()

        # =========  VAL RESPONSES (20 trials per stim, per-site stitching)  ===========

        val_files = sorted(os.listdir(os.path.join(path, f'{area}_single_sites')))

        # Accumulate (S_val, R, N_subpop, T) per site, then stitch across cells.
        per_site_val = []
        val_cells_in_order = []
        for filename in tqdm(val_files, desc=f'NAT4 {area} val sites'):
            # 'TNC*' sites do not have est data, but they DO have val data.
            # We still want their val responses stitched in — they appear
            # in the population's chans list and est-set rows are NaN-sentinels.
            datafile = os.path.join(path, f'{area}_single_sites', filename)
            single_site_rec = load_recording(datafile)
            val_cells_in_order += single_site_rec['resp'].chans

            site_responses = []
            for val_sound in val_sounds:
                resp = single_site_rec['resp'].rasterize()
                arr = resp.extract_epoch(val_sound)   # (R=20, N_subpop, T_ms=1500)
                R, N_subpop, T_ms = arr.shape
                # downsample 1 ms -> 10 ms by summing
                arr = arr.reshape(R, N_subpop, -1, 10).sum(axis=-1)
                site_responses.append(torch.from_numpy(arr))
                del resp
            per_site_val.append(torch.stack(site_responses))  # (S_val, R, N_subpop, T)
            del single_site_rec
            gc.collect()

        val_full = torch.cat(per_site_val, dim=2)         # (S_val, R, N_total_in_val_order, T)
        val_full = val_full.permute(0, 2, 1, 3)           # (S_val, N, R, T)

        # Cells in the val concatenation are not in the same order as the
        # population's `cells` list, but the SET is the same. Reindex.
        index_map = {u: i for i, u in enumerate(val_cells_in_order)}
        # Cells in the population list that are NOT in val_cells_in_order
        # were never presented val stims (e.g. some TNC cells were
        # est-only) — give them NaN sentinels.
        S_val = val_full.shape[0]
        val_responses_per_stim = []
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
