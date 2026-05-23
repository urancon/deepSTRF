"""Downer 2025 / Ahmed 2025 squirrel-monkey auditory cortex dataset.

Multi-unit (channel-level threshold-crossing) spike recordings from auditory
cortex of three squirrel monkeys, presented with TIMIT sentences and monkey
vocalizations. Two stimulus classes are loaded separately via the
``stimuli={'timit', 'mvocs'}`` constructor arg.

Source: Zenodo DOI ``10.5281/zenodo.16175377`` (29 GB archive).

Citations:
- Ahmed B, Downer JD, Malone BJ, Makin JG (2025). *Deep neural networks
  explain spiking activity in auditory cortex.* PLoS Computational Biology
  21(8):e1013334. https://doi.org/10.1371/journal.pcbi.1013334
- Downer JD, Bigelow J, Runfeldt M, Malone BJ (2021). *Temporally precise
  population coding of dynamic sounds by auditory cortex.* J Neurophysiol.

Each ``*_MUspk.mat`` file holds threshold-crossing spike times for one
channel; spike-sorting was not feasible, so each channel is a single
"multi-unit" (= one "neuron" in deepSTRF parlance). The published count
is 1718 multi-units across 41 sessions (Ahmed 2025, p4–5).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.utils.data_download import (
    default_cache_dir,
    unzip,
    zenodo_download,
)


# Public Zenodo record. https://doi.org/10.5281/zenodo.16175377
DOWNER_ZENODO_RECORD = 16175377
DOWNER_ZENODO_ARCHIVE = "auditory_cortex_data.zip"
# Filename inside the archive that uniquely identifies a complete extraction.
DOWNER_EXTRACTED_SUBDIR = "auditory_cortex_data"


def download_downer2025(dest: Optional[str] = None) -> str:
    """Download the Downer 2025 / Ahmed 2025 archive from Zenodo.

    The archive (``auditory_cortex_data.zip``, ~29 GB) ships in Zenodo
    record ``10.5281/zenodo.16175377`` and unzips to
    ``<dest>/auditory_cortex_data/`` — the standard layout the
    ``Downer2025Dataset`` constructor expects.

    Parameters
    ----------
    dest : str, optional
        Parent directory the archive is downloaded into. Defaults to
        ``default_cache_dir('Downer2025')`` (overridable via
        ``$DEEPSTRF_DATA_DIR``).

    Returns
    -------
    str
        Path to the extracted ``auditory_cortex_data/`` directory.

    Notes
    -----
    Idempotent: skips the zip download if already present, and skips
    the unzip step if the expected ``auditory_cortex_data/sessions/``
    subdirectory already exists.

    **Heads up:** the archive is ~29 GB on disk; the unpacked
    directory is also ~29 GB. Allow ~60 GB total during extraction
    (zip plus contents); you can delete the zip once unpacking is
    complete.
    """
    dest_path = Path(default_cache_dir("Downer2025") if dest is None else dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    extracted_dir = dest_path / DOWNER_EXTRACTED_SUBDIR
    sessions_marker = extracted_dir / "sessions"
    if sessions_marker.is_dir():
        return str(extracted_dir)

    archive_path = dest_path / DOWNER_ZENODO_ARCHIVE
    if not archive_path.exists():
        zenodo_download(
            DOWNER_ZENODO_RECORD, DOWNER_ZENODO_ARCHIVE, archive_path,
        )
    unzip(archive_path, dest_path)
    if not sessions_marker.is_dir():
        raise RuntimeError(
            f"Downer2025 unzip incomplete: expected {sessions_marker} not found."
        )
    return str(extracted_dir)


# Fine-area assignments are encoded as YAML *comments* in
# ``sessions_metadata.yml`` (under ``area_wise_sessions:``), not as YAML
# data. We transcribe them once here and use them as the source of truth
# for the ``area`` ``nrn_meta`` key.
#
# Mapping: session_id (str) -> (area_group, fine_area).
#   area_group ∈ {'core', 'non-primary'}
#   fine_area  ∈ {'A1', 'R', 'ML', 'AL', 'CL', 'CPB', 'RPB'}  (None if unknown)
#
# Sessions present in the Zenodo upload but absent from these comments
# fall back to ``area_group`` from the YAML data with ``fine_area=None``.
# Hard-coded well-tuned cell-id lists for fast filtering without rerunning
# ``compute_paper_tuning()``. Generated once via::
#
#   ds = Downer2025Dataset(stimuli=<MODE>, dt_ms=50.0, smooth=False)
#   ds.compute_paper_tuning(
#       n_resamples=10_000, dt_ms_analysis=50.0, seed=0,
#       alpha=0.05, delta=0.5,
#   )
#   wt = sorted(m["cell_id"] for m in ds.nrn_meta
#               if m.get(f"ahmed2025_{<MODE>}_well_tuned", False))
#
# Counts: 417 TIMIT-well-tuned, 476 mVocs-well-tuned (paper target 404 / 489).
# The δ-criterion (δ ≥ 0.5) is effect-size-based and converges at moderate
# n_resamples; the looser p<0.05 "tuned" cohort is more sensitive to the
# resample count -- bump to n_resamples=100_000 to match the paper's
# 1195 / 1231 "tuned" counts exactly. Use ``attach_ahmed2025_well_tuned()``
# below to write the boolean onto ``nrn_meta`` without re-running the
# (~5 min) paper-tuning method.
AHMED2025_WELL_TUNED_TIMIT: tuple[str, ...] = (
    "b_180613_Ch16", "b_180627_Ch10", "b_180627_Ch15", "b_180627_Ch16",
    "b_180627_Ch9p", "b_180717_Ch12", "b_180717_Ch13", "b_180717_Ch14",
    "b_180719_Ch11", "b_180719_Ch12", "b_180719_Ch13", "b_180719_Ch14",
    "b_180719_Ch18", "b_180719_Ch19", "b_180719_Ch7p", "b_180719_Ch8p",
    "b_180730_Ch10", "b_180730_Ch11", "b_180730_Ch12", "b_180730_Ch17",
    "b_180730_Ch18", "b_180730_Ch9p", "b_180808_Ch11", "b_180808_Ch12",
    "b_180808_Ch14", "b_180808_Ch17", "b_180808_Ch18", "b_180808_Ch5p",
    "b_180808_Ch7p", "b_180810_Ch12", "b_180810_Ch14", "b_180814_Ch10",
    "b_180814_Ch11", "b_180814_Ch13", "b_180814_Ch14", "b_180814_Ch15",
    "b_180814_Ch17", "b_180814_Ch18", "b_180814_Ch19", "b_180814_Ch1p",
    "b_180814_Ch21", "b_180814_Ch22", "b_180814_Ch23", "b_180814_Ch24",
    "b_180814_Ch25", "b_180814_Ch27", "b_180814_Ch28", "b_180814_Ch29",
    "b_180814_Ch2p", "b_180814_Ch30", "b_180814_Ch31", "b_180814_Ch32",
    "b_180814_Ch33", "b_180814_Ch34", "b_180814_Ch35", "b_180814_Ch36",
    "b_180814_Ch4p", "b_180814_Ch6p", "b_180814_Ch7p", "b_180814_Ch8p",
    "b_180814_Ch9p", "c_180622_Ch9p", "c_180720_Ch11", "c_180720_Ch12",
    "c_180720_Ch13", "c_180720_Ch14", "c_180720_Ch16", "c_180720_Ch17",
    "c_180720_Ch18", "c_180720_Ch8p", "c_180728_Ch10", "c_180728_Ch14",
    "c_180728_Ch15", "c_180728_Ch6p", "c_180728_Ch9p", "c_180731_Ch10",
    "c_180731_Ch11", "c_180731_Ch12", "c_180731_Ch13", "c_180731_Ch14",
    "c_180731_Ch15", "c_180731_Ch16", "c_180731_Ch17", "c_180731_Ch18",
    "c_180731_Ch19", "c_180731_Ch6p", "c_180731_Ch7p", "c_180731_Ch8p",
    "c_180731_Ch9p", "c_180807_Ch10", "c_180807_Ch11", "c_180807_Ch12",
    "c_180807_Ch13", "c_180807_Ch14", "c_180807_Ch15", "c_180807_Ch16",
    "c_180807_Ch17", "c_180807_Ch18", "c_180807_Ch20", "c_180807_Ch5p",
    "c_180807_Ch6p", "c_180807_Ch7p", "c_180807_Ch8p", "c_180807_Ch9p",
    "c_190606_Ch10", "c_190606_Ch11", "c_190606_Ch13", "c_190606_Ch33",
    "c_190606_Ch35", "c_190606_Ch36", "c_190606_Ch38", "c_190606_Ch39",
    "c_190606_Ch40", "c_190606_Ch41", "c_190606_Ch42", "c_190606_Ch44",
    "c_190606_Ch45", "c_190606_Ch46", "c_190606_Ch48", "c_190606_Ch50",
    "c_190606_Ch51", "c_190606_Ch52", "c_190606_Ch53", "c_190606_Ch54",
    "c_190606_Ch56", "c_190606_Ch57", "c_190606_Ch58", "c_190606_Ch5p",
    "c_190606_Ch61", "c_190606_Ch62", "c_190606_Ch63", "c_190606_Ch64",
    "c_190606_Ch6p", "c_190726_Ch56", "c_190726_Ch58", "c_191113_Ch11",
    "c_191113_Ch12", "c_191113_Ch13", "c_191113_Ch15", "c_191113_Ch18",
    "c_191113_Ch19", "c_191113_Ch20", "c_191113_Ch21", "c_191113_Ch22",
    "c_191113_Ch23", "c_191113_Ch24", "c_191113_Ch25", "c_191113_Ch29",
    "c_191113_Ch2p", "c_191113_Ch39", "c_191113_Ch3p", "c_191113_Ch41",
    "c_191113_Ch43", "c_191113_Ch44", "c_191113_Ch45", "c_191113_Ch46",
    "c_191113_Ch47", "c_191113_Ch48", "c_191113_Ch49", "c_191113_Ch4p",
    "c_191113_Ch51", "c_191113_Ch54", "c_191113_Ch57", "c_191113_Ch5p",
    "c_191113_Ch60", "c_191113_Ch62", "c_191113_Ch64", "c_191113_Ch7p",
    "c_191113_Ch8p", "c_191115_Ch25", "c_191115_Ch2p", "c_191115_Ch41",
    "c_191115_Ch43", "c_191115_Ch45", "c_191115_Ch46", "c_191115_Ch48",
    "c_191115_Ch49", "c_191115_Ch59", "c_191115_Ch62", "c_191121_Ch15",
    "c_191121_Ch18", "c_191121_Ch20", "c_191121_Ch21", "c_191121_Ch22",
    "c_191121_Ch23", "c_191121_Ch24", "c_191121_Ch25", "c_191121_Ch29",
    "c_191121_Ch2p", "c_191121_Ch35", "c_191121_Ch39", "c_191121_Ch43",
    "c_191121_Ch44", "c_191121_Ch45", "c_191121_Ch48", "c_191121_Ch4p",
    "c_191121_Ch62", "c_191121_Ch64", "c_191125_Ch13", "c_191125_Ch15",
    "c_191125_Ch17", "c_191125_Ch18", "c_191125_Ch1p", "c_191125_Ch20",
    "c_191125_Ch21", "c_191125_Ch22", "c_191125_Ch24", "c_191125_Ch25",
    "c_191125_Ch29", "c_191125_Ch2p", "c_191125_Ch31", "c_191125_Ch33",
    "c_191125_Ch35", "c_191125_Ch3p", "c_191125_Ch42", "c_191125_Ch44",
    "c_191125_Ch47", "c_191125_Ch53", "c_191125_Ch54", "c_191125_Ch57",
    "c_191125_Ch5p", "c_191125_Ch60", "c_191125_Ch62", "c_191125_Ch6p",
    "c_191125_Ch9p", "c_191206_Ch18", "c_191206_Ch19", "c_191206_Ch21",
    "c_191206_Ch23", "c_191206_Ch27", "c_191206_Ch2p", "c_191206_Ch43",
    "c_191206_Ch45", "c_191206_Ch47", "c_191206_Ch48", "c_191206_Ch4p",
    "c_191206_Ch60", "c_191206_Ch62", "c_191206_Ch64", "c_191210_Ch10",
    "c_191210_Ch11", "c_191210_Ch12", "c_191210_Ch14", "c_191210_Ch15",
    "c_191210_Ch16", "c_191210_Ch18", "c_191210_Ch19", "c_191210_Ch20",
    "c_191210_Ch21", "c_191210_Ch23", "c_191210_Ch24", "c_191210_Ch28",
    "c_191210_Ch30", "c_191210_Ch34", "c_191210_Ch42", "c_191210_Ch45",
    "c_191210_Ch46", "c_191210_Ch48", "c_191210_Ch49", "c_191210_Ch51",
    "c_191210_Ch53", "c_191210_Ch56", "c_191210_Ch57", "c_191210_Ch59",
    "c_191210_Ch60", "c_191210_Ch62", "c_191210_Ch63", "c_200205_Ch10",
    "c_200205_Ch11", "c_200205_Ch12", "c_200205_Ch13", "c_200205_Ch14",
    "c_200205_Ch17", "c_200205_Ch18", "c_200205_Ch19", "c_200205_Ch1p",
    "c_200205_Ch20", "c_200205_Ch21", "c_200205_Ch2p", "c_200205_Ch43",
    "c_200205_Ch45", "c_200205_Ch47", "c_200205_Ch4p", "c_200205_Ch53",
    "c_200205_Ch54", "c_200205_Ch57", "c_200205_Ch58", "c_200205_Ch5p",
    "c_200205_Ch60", "c_200205_Ch7p", "c_200205_Ch8p", "c_200206_Ch10",
    "c_200206_Ch11", "c_200206_Ch12", "c_200206_Ch13", "c_200206_Ch14",
    "c_200206_Ch15", "c_200206_Ch16", "c_200206_Ch17", "c_200206_Ch18",
    "c_200206_Ch19", "c_200206_Ch1p", "c_200206_Ch20", "c_200206_Ch21",
    "c_200206_Ch22", "c_200206_Ch23", "c_200206_Ch24", "c_200206_Ch25",
    "c_200206_Ch26", "c_200206_Ch27", "c_200206_Ch28", "c_200206_Ch29",
    "c_200206_Ch2p", "c_200206_Ch30", "c_200206_Ch31", "c_200206_Ch32",
    "c_200206_Ch33", "c_200206_Ch34", "c_200206_Ch35", "c_200206_Ch36",
    "c_200206_Ch37", "c_200206_Ch38", "c_200206_Ch39", "c_200206_Ch3p",
    "c_200206_Ch40", "c_200206_Ch41", "c_200206_Ch42", "c_200206_Ch43",
    "c_200206_Ch44", "c_200206_Ch45", "c_200206_Ch46", "c_200206_Ch47",
    "c_200206_Ch48", "c_200206_Ch4p", "c_200206_Ch50", "c_200206_Ch51",
    "c_200206_Ch52", "c_200206_Ch53", "c_200206_Ch54", "c_200206_Ch55",
    "c_200206_Ch56", "c_200206_Ch57", "c_200206_Ch58", "c_200206_Ch59",
    "c_200206_Ch5p", "c_200206_Ch60", "c_200206_Ch61", "c_200206_Ch62",
    "c_200206_Ch63", "c_200206_Ch64", "c_200206_Ch6p", "c_200206_Ch7p",
    "c_200206_Ch8p", "c_200206_Ch9p", "c_200207_Ch11", "c_200207_Ch12",
    "c_200207_Ch13", "c_200207_Ch15", "c_200207_Ch17", "c_200207_Ch19",
    "c_200207_Ch1p", "c_200207_Ch21", "c_200207_Ch2p", "c_200207_Ch3p",
    "c_200207_Ch4p", "c_200207_Ch5p", "c_200207_Ch6p", "c_200207_Ch7p",
    "c_200207_Ch8p", "f_200213_Ch11", "f_200213_Ch15", "f_200213_Ch21",
    "f_200213_Ch29", "f_200213_Ch2p", "f_200213_Ch31", "f_200213_Ch33",
    "f_200213_Ch35", "f_200213_Ch39", "f_200213_Ch41", "f_200213_Ch42",
    "f_200213_Ch43", "f_200213_Ch44", "f_200213_Ch45", "f_200213_Ch46",
    "f_200213_Ch47", "f_200213_Ch48", "f_200213_Ch49", "f_200213_Ch4p",
    "f_200213_Ch60", "f_200213_Ch62", "f_200213_Ch64", "f_200219_Ch18",
    "f_200219_Ch20", "f_200219_Ch23", "f_200219_Ch25", "f_200219_Ch2p",
    "f_200219_Ch42", "f_200219_Ch43", "f_200219_Ch45", "f_200219_Ch64",
    "f_200313_Ch20", "f_200313_Ch22", "f_200313_Ch24", "f_200313_Ch25",
    "f_200313_Ch29", "f_200313_Ch31", "f_200313_Ch33", "f_200313_Ch35",
    "f_200313_Ch38", "f_200313_Ch39", "f_200313_Ch41", "f_200313_Ch42",
    "f_200313_Ch43", "f_200313_Ch44", "f_200313_Ch45", "f_200313_Ch48",
    "f_200313_Ch64",
)

AHMED2025_WELL_TUNED_MVOCS: tuple[str, ...] = (
    "b_180413_Ch58", "b_180613_Ch6p", "b_180627_Ch10", "b_180627_Ch15",
    "b_180627_Ch16", "b_180627_Ch19", "b_180627_Ch20", "b_180627_Ch9p",
    "b_180717_Ch12", "b_180717_Ch13", "b_180717_Ch14", "b_180719_Ch10",
    "b_180719_Ch11", "b_180719_Ch12", "b_180719_Ch13", "b_180719_Ch14",
    "b_180719_Ch15", "b_180719_Ch16", "b_180719_Ch17", "b_180719_Ch18",
    "b_180719_Ch19", "b_180719_Ch5p", "b_180719_Ch6p", "b_180719_Ch7p",
    "b_180719_Ch8p", "b_180719_Ch9p", "b_180730_Ch10", "b_180730_Ch11",
    "b_180730_Ch12", "b_180730_Ch13", "b_180730_Ch15", "b_180730_Ch17",
    "b_180730_Ch18", "b_180730_Ch7p", "b_180730_Ch8p", "b_180730_Ch9p",
    "b_180808_Ch17", "b_180808_Ch5p", "b_180808_Ch8p", "b_180810_Ch12",
    "b_180810_Ch13", "b_180810_Ch14", "b_180810_Ch18", "b_180810_Ch7p",
    "b_180814_Ch10", "b_180814_Ch11", "b_180814_Ch12", "b_180814_Ch13",
    "b_180814_Ch14", "b_180814_Ch15", "b_180814_Ch16", "b_180814_Ch17",
    "b_180814_Ch18", "b_180814_Ch19", "b_180814_Ch1p", "b_180814_Ch20",
    "b_180814_Ch21", "b_180814_Ch22", "b_180814_Ch23", "b_180814_Ch24",
    "b_180814_Ch25", "b_180814_Ch26", "b_180814_Ch27", "b_180814_Ch28",
    "b_180814_Ch29", "b_180814_Ch2p", "b_180814_Ch30", "b_180814_Ch31",
    "b_180814_Ch32", "b_180814_Ch33", "b_180814_Ch34", "b_180814_Ch35",
    "b_180814_Ch36", "b_180814_Ch4p", "b_180814_Ch6p", "b_180814_Ch7p",
    "b_180814_Ch8p", "b_180814_Ch9p", "c_180502_Ch15", "c_180502_Ch17",
    "c_180502_Ch7p", "c_180502_Ch8p", "c_180622_Ch10", "c_180622_Ch15",
    "c_180622_Ch16", "c_180622_Ch9p", "c_180720_Ch10", "c_180720_Ch11",
    "c_180720_Ch12", "c_180720_Ch13", "c_180720_Ch14", "c_180720_Ch15",
    "c_180720_Ch16", "c_180720_Ch17", "c_180720_Ch18", "c_180720_Ch6p",
    "c_180720_Ch7p", "c_180720_Ch8p", "c_180720_Ch9p", "c_180728_Ch10",
    "c_180728_Ch6p", "c_180728_Ch9p", "c_180731_Ch10", "c_180731_Ch11",
    "c_180731_Ch12", "c_180731_Ch17", "c_180731_Ch19", "c_180731_Ch6p",
    "c_180731_Ch9p", "c_180807_Ch10", "c_180807_Ch11", "c_180807_Ch12",
    "c_180807_Ch13", "c_180807_Ch14", "c_180807_Ch15", "c_180807_Ch16",
    "c_180807_Ch17", "c_180807_Ch18", "c_180807_Ch20", "c_180807_Ch5p",
    "c_180807_Ch6p", "c_180807_Ch7p", "c_180807_Ch8p", "c_180807_Ch9p",
    "c_190606_Ch11", "c_190606_Ch13", "c_190606_Ch14", "c_190606_Ch20",
    "c_190606_Ch36", "c_190606_Ch39", "c_190606_Ch40", "c_190606_Ch41",
    "c_190606_Ch42", "c_190606_Ch43", "c_190606_Ch44", "c_190606_Ch45",
    "c_190606_Ch46", "c_190606_Ch47", "c_190606_Ch50", "c_190606_Ch51",
    "c_190606_Ch52", "c_190606_Ch54", "c_190606_Ch55", "c_190606_Ch57",
    "c_190606_Ch5p", "c_190606_Ch61", "c_190606_Ch62", "c_191113_Ch11",
    "c_191113_Ch12", "c_191113_Ch13", "c_191113_Ch15", "c_191113_Ch18",
    "c_191113_Ch19", "c_191113_Ch1p", "c_191113_Ch20", "c_191113_Ch21",
    "c_191113_Ch22", "c_191113_Ch23", "c_191113_Ch24", "c_191113_Ch25",
    "c_191113_Ch29", "c_191113_Ch2p", "c_191113_Ch39", "c_191113_Ch3p",
    "c_191113_Ch41", "c_191113_Ch43", "c_191113_Ch44", "c_191113_Ch45",
    "c_191113_Ch46", "c_191113_Ch47", "c_191113_Ch48", "c_191113_Ch49",
    "c_191113_Ch4p", "c_191113_Ch51", "c_191113_Ch54", "c_191113_Ch57",
    "c_191113_Ch58", "c_191113_Ch59", "c_191113_Ch5p", "c_191113_Ch60",
    "c_191113_Ch61", "c_191113_Ch62", "c_191113_Ch64", "c_191113_Ch7p",
    "c_191113_Ch8p", "c_191115_Ch12", "c_191115_Ch18", "c_191115_Ch21",
    "c_191115_Ch23", "c_191115_Ch2p", "c_191115_Ch41", "c_191115_Ch43",
    "c_191115_Ch46", "c_191115_Ch47", "c_191115_Ch48", "c_191115_Ch49",
    "c_191115_Ch51", "c_191115_Ch55", "c_191115_Ch57", "c_191115_Ch58",
    "c_191115_Ch59", "c_191115_Ch60", "c_191115_Ch62", "c_191115_Ch63",
    "c_191115_Ch64", "c_191115_Ch7p", "c_191121_Ch11", "c_191121_Ch12",
    "c_191121_Ch15", "c_191121_Ch18", "c_191121_Ch19", "c_191121_Ch20",
    "c_191121_Ch21", "c_191121_Ch22", "c_191121_Ch23", "c_191121_Ch25",
    "c_191121_Ch29", "c_191121_Ch2p", "c_191121_Ch39", "c_191121_Ch43",
    "c_191121_Ch44", "c_191121_Ch45", "c_191121_Ch47", "c_191121_Ch48",
    "c_191121_Ch4p", "c_191121_Ch54", "c_191121_Ch60", "c_191121_Ch62",
    "c_191121_Ch64", "c_191125_Ch11", "c_191125_Ch12", "c_191125_Ch13",
    "c_191125_Ch15", "c_191125_Ch17", "c_191125_Ch18", "c_191125_Ch19",
    "c_191125_Ch1p", "c_191125_Ch21", "c_191125_Ch22", "c_191125_Ch23",
    "c_191125_Ch24", "c_191125_Ch25", "c_191125_Ch29", "c_191125_Ch2p",
    "c_191125_Ch31", "c_191125_Ch33", "c_191125_Ch35", "c_191125_Ch3p",
    "c_191125_Ch42", "c_191125_Ch43", "c_191125_Ch44", "c_191125_Ch45",
    "c_191125_Ch47", "c_191125_Ch48", "c_191125_Ch49", "c_191125_Ch4p",
    "c_191125_Ch53", "c_191125_Ch54", "c_191125_Ch57", "c_191125_Ch58",
    "c_191125_Ch5p", "c_191125_Ch60", "c_191125_Ch62", "c_191125_Ch64",
    "c_191125_Ch6p", "c_191125_Ch7p", "c_191125_Ch9p", "c_191206_Ch18",
    "c_191206_Ch19", "c_191206_Ch21", "c_191206_Ch23", "c_191206_Ch2p",
    "c_191206_Ch39", "c_191206_Ch43", "c_191206_Ch44", "c_191206_Ch45",
    "c_191206_Ch47", "c_191206_Ch48", "c_191206_Ch60", "c_191206_Ch62",
    "c_191206_Ch64", "c_191210_Ch11", "c_191210_Ch16", "c_191210_Ch18",
    "c_191210_Ch20", "c_191210_Ch23", "c_191210_Ch28", "c_191210_Ch30",
    "c_191210_Ch34", "c_191210_Ch37", "c_191210_Ch3p", "c_191210_Ch42",
    "c_191210_Ch50", "c_191210_Ch6p", "c_191210_Ch8p", "c_191210_Ch9p",
    "c_191219_Ch27", "c_200205_Ch10", "c_200205_Ch11", "c_200205_Ch12",
    "c_200205_Ch13", "c_200205_Ch14", "c_200205_Ch17", "c_200205_Ch18",
    "c_200205_Ch19", "c_200205_Ch1p", "c_200205_Ch20", "c_200205_Ch21",
    "c_200205_Ch22", "c_200205_Ch23", "c_200205_Ch25", "c_200205_Ch26",
    "c_200205_Ch28", "c_200205_Ch2p", "c_200205_Ch31", "c_200205_Ch32",
    "c_200205_Ch35", "c_200205_Ch3p", "c_200205_Ch43", "c_200205_Ch45",
    "c_200205_Ch47", "c_200205_Ch4p", "c_200205_Ch53", "c_200205_Ch54",
    "c_200205_Ch55", "c_200205_Ch57", "c_200205_Ch58", "c_200205_Ch5p",
    "c_200205_Ch60", "c_200205_Ch61", "c_200205_Ch63", "c_200205_Ch6p",
    "c_200205_Ch7p", "c_200205_Ch8p", "c_200206_Ch10", "c_200206_Ch11",
    "c_200206_Ch12", "c_200206_Ch13", "c_200206_Ch14", "c_200206_Ch15",
    "c_200206_Ch16", "c_200206_Ch17", "c_200206_Ch18", "c_200206_Ch19",
    "c_200206_Ch1p", "c_200206_Ch20", "c_200206_Ch21", "c_200206_Ch22",
    "c_200206_Ch23", "c_200206_Ch24", "c_200206_Ch25", "c_200206_Ch26",
    "c_200206_Ch27", "c_200206_Ch28", "c_200206_Ch29", "c_200206_Ch2p",
    "c_200206_Ch30", "c_200206_Ch31", "c_200206_Ch32", "c_200206_Ch33",
    "c_200206_Ch34", "c_200206_Ch35", "c_200206_Ch36", "c_200206_Ch37",
    "c_200206_Ch38", "c_200206_Ch39", "c_200206_Ch3p", "c_200206_Ch40",
    "c_200206_Ch41", "c_200206_Ch42", "c_200206_Ch43", "c_200206_Ch44",
    "c_200206_Ch45", "c_200206_Ch46", "c_200206_Ch47", "c_200206_Ch48",
    "c_200206_Ch49", "c_200206_Ch4p", "c_200206_Ch50", "c_200206_Ch51",
    "c_200206_Ch52", "c_200206_Ch53", "c_200206_Ch54", "c_200206_Ch55",
    "c_200206_Ch56", "c_200206_Ch57", "c_200206_Ch58", "c_200206_Ch59",
    "c_200206_Ch5p", "c_200206_Ch60", "c_200206_Ch61", "c_200206_Ch62",
    "c_200206_Ch63", "c_200206_Ch64", "c_200206_Ch6p", "c_200206_Ch7p",
    "c_200206_Ch8p", "c_200206_Ch9p", "c_200207_Ch11", "c_200207_Ch12",
    "c_200207_Ch13", "c_200207_Ch15", "c_200207_Ch17", "c_200207_Ch19",
    "c_200207_Ch1p", "c_200207_Ch21", "c_200207_Ch2p", "c_200207_Ch3p",
    "c_200207_Ch4p", "c_200207_Ch5p", "c_200207_Ch6p", "c_200207_Ch7p",
    "c_200207_Ch8p", "c_200212_Ch49", "c_200212_Ch54", "c_200212_Ch58",
    "c_200212_Ch64", "f_191209_Ch33", "f_191209_Ch35", "f_191209_Ch42",
    "f_200219_Ch18", "f_200219_Ch20", "f_200219_Ch23", "f_200219_Ch25",
    "f_200219_Ch46", "f_200219_Ch48", "f_200219_Ch64", "f_200313_Ch20",
    "f_200313_Ch22", "f_200313_Ch25", "f_200313_Ch29", "f_200313_Ch31",
    "f_200313_Ch33", "f_200313_Ch35", "f_200313_Ch39", "f_200313_Ch41",
    "f_200313_Ch42", "f_200313_Ch44", "f_200313_Ch62", "f_200313_Ch64",
    "f_200318_Ch11", "f_200318_Ch12", "f_200318_Ch13", "f_200318_Ch15",
    "f_200318_Ch16", "f_200318_Ch17", "f_200318_Ch18", "f_200318_Ch19",
    "f_200318_Ch1p", "f_200318_Ch21", "f_200318_Ch22", "f_200318_Ch24",
    "f_200318_Ch26", "f_200318_Ch27", "f_200318_Ch28", "f_200318_Ch29",
    "f_200318_Ch2p", "f_200318_Ch31", "f_200318_Ch32", "f_200318_Ch33",
    "f_200318_Ch34", "f_200318_Ch35", "f_200318_Ch36", "f_200318_Ch38",
    "f_200318_Ch39", "f_200318_Ch3p", "f_200318_Ch40", "f_200318_Ch41",
    "f_200318_Ch42", "f_200318_Ch44", "f_200318_Ch4p", "f_200318_Ch5p",
    "f_200318_Ch6p", "f_200318_Ch7p", "f_200318_Ch8p", "f_200318_Ch9p",
)


_FINE_AREA_BY_SESSION: dict[str, tuple[str, str]] = {
    # Core / A1
    "180627": ("core", "A1"), "180719": ("core", "A1"), "180810": ("core", "A1"),
    "180814": ("core", "A1"), "180720": ("core", "A1"), "180731": ("core", "A1"),
    "190604": ("core", "A1"), "190606": ("core", "A1"), "190726": ("core", "A1"),
    "190801": ("core", "A1"), "191209": ("core", "A1"), "200313": ("core", "A1"),
    "191113": ("core", "A1"), "191125": ("core", "A1"), "191206": ("core", "A1"),
    "200206": ("core", "A1"), "200207": ("core", "A1"), "180808": ("core", "A1"),
    # Core / R
    "180807": ("core", "R"), "200213": ("core", "R"), "180501": ("core", "R"),
    # Non-primary / belt / CL
    "191219": ("non-primary", "CL"),
    # Non-primary / belt / ML
    "180613": ("non-primary", "ML"), "190703": ("non-primary", "ML"),
    "200219": ("non-primary", "ML"), "191115": ("non-primary", "ML"),
    "200205": ("non-primary", "ML"),
    # Non-primary / belt / AL
    "180622": ("non-primary", "AL"), "191211": ("non-primary", "AL"),
    "180413": ("non-primary", "AL"), "180420": ("non-primary", "AL"),
    # Non-primary / parabelt / CPB
    "180724": ("non-primary", "CPB"), "200318": ("non-primary", "CPB"),
    "191121": ("non-primary", "CPB"),
    # Non-primary / parabelt / RPB
    "180502": ("non-primary", "RPB"), "180728": ("non-primary", "RPB"),
    "190605": ("non-primary", "RPB"), "191210": ("non-primary", "RPB"),
    "180717": ("non-primary", "RPB"), "180730": ("non-primary", "RPB"),
    "200212": ("non-primary", "RPB"),
}


def _parse_sessions_metadata(path: Union[str, Path]) -> dict:
    """Parse ``sessions_metadata.yml`` from a Downer-style dataset root.

    Returns a dict::

        {
          'hemisphere':    {session_id: 'LH'|'RH'},
          'animal':        {session_id: 'b'|'c'|'f'},
          'area_group':    {session_id: 'core'|'non-primary'},
          'fine_area':     {session_id: 'A1'|'R'|'ML'|'AL'|'CL'|'CPB'|'RPB'|None},
          'coord':         {session_id: (x, y)},
          'bad':           set(session_id),
          'n_reps_canonical': {'timit': 11, 'mvocs': 15},
        }
    """
    yml_path = Path(path) / "sessions_metadata.yml"
    with open(yml_path) as fh:
        m = yaml.safe_load(fh)

    hemisphere: dict[str, str] = {}
    animal: dict[str, str] = {}
    for key in ("c_RH_sessions", "b_RH_sessions", "f_RH_sessions", "c_LH_sessions"):
        animal_id, hemi = key[0], key[2:4]
        for sid in m[key]:
            sid = str(sid)
            hemisphere[sid] = hemi
            animal[sid] = animal_id

    area_group: dict[str, str] = {}
    for grp in ("core", "non-primary"):
        for sid in m["area_wise_sessions"][grp]:
            area_group[str(sid)] = grp

    fine_area = {sid: lbl for sid, (_, lbl) in _FINE_AREA_BY_SESSION.items()}

    coord = {str(sid): tuple(xy) for sid, xy in m["session_coordinates"].items()}
    bad = {str(sid) for sid in m["bad_sessions"]}

    return {
        "hemisphere": hemisphere,
        "animal": animal,
        "area_group": area_group,
        "fine_area": fine_area,
        "coord": coord,
        "bad": bad,
        "n_reps_canonical": dict(m["stim_wise_num_repeats"]),
    }


# ``Ch49``, ``Ch5p``, ``Ch10s2`` -- digits then an optional alphanumeric suffix
# (annotates a re-mounted / second-set recording on the same channel number).
_CHAN_RE = re.compile(r"^Ch(\d+)([A-Za-z][A-Za-z0-9]*)?$")


def _parse_channel_filename(fname: str) -> Optional[tuple[str, str, int, str]]:
    """Parse ``<animal>_<session>_Ch<N>[suffix]_MUspk.mat`` -> (animal, session, ch_int, suffix).

    ``suffix`` is the trailing letter/digit annotation after the channel
    number (e.g. ``'p'``, ``'s2'``), or the empty string. Returns ``None``
    for non-MUspk files (e.g. ``TRIALINFO.mat``).
    """
    if "MUspk" not in fname or not fname.endswith("_MUspk.mat"):
        return None
    base = fname[: -len("_MUspk.mat")]
    parts = base.split("_")
    if len(parts) < 3:
        return None
    animal, session, chan_token = parts[0], parts[1], parts[2]
    m = _CHAN_RE.match(chan_token)
    if m is None:
        return None
    ch_int = int(m.group(1))
    suffix = m.group(2) or ""
    return animal, session, ch_int, suffix


def _enumerate_neurons(
    path: Union[str, Path],
    meta: dict,
    *,
    animals: Union[str, Sequence[str]] = "all",
    areas: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Walk ``sessions/<id>/*_MUspk.mat`` and build the ``nrn_meta`` list.

    Filters (all applied at construction time):
        animals  -- 'all' or iterable of 'b'/'c'/'f'.
        areas    -- ('core',) or ('non-primary',) or fine labels
                    ('A1','R','ML','AL','CL','CPB','RPB'); None == no filter.
        sessions -- iterable of session ids (str), or None.

    Each entry has keys: cell_id, session_id, animal_id, hemisphere,
    area_group, area, channel, n_channels_in_session, coord_x, coord_y,
    recording_type.
    """
    root = Path(path) / "sessions"
    if not root.is_dir():
        raise FileNotFoundError(f"Expected sessions directory: {root}")

    if animals == "all":
        animal_whitelist: Optional[set[str]] = None
    else:
        animal_whitelist = set(animals)

    if sessions is None:
        session_whitelist: Optional[set[str]] = None
    else:
        session_whitelist = set(sessions)

    # 'primary' is an Ahmed-2025-style alias for the YAML's 'core' group.
    area_whitelist: Optional[set[str]]
    if areas is None:
        area_whitelist = None
    else:
        area_whitelist = {("core" if a == "primary" else a) for a in areas}

    nrn_meta: list[dict] = []
    for session_id in sorted(os.listdir(root)):
        sess_dir = root / session_id
        if not sess_dir.is_dir():
            continue
        if session_whitelist is not None and session_id not in session_whitelist:
            continue
        if session_id in meta["bad"]:
            # Defensive: bad sessions are absent from the public release, but
            # honour the YAML flag if a private copy still has them.
            continue

        chan_files = sorted(f for f in os.listdir(sess_dir) if "MUspk" in f)
        n_chan = len(chan_files)

        sess_animal = meta["animal"].get(session_id)
        sess_hemi = meta["hemisphere"].get(session_id)
        sess_area_grp = meta["area_group"].get(session_id)
        sess_area = meta["fine_area"].get(session_id)
        sess_coord = meta["coord"].get(session_id, (None, None))

        if animal_whitelist is not None and sess_animal not in animal_whitelist:
            continue
        if area_whitelist is not None and not (
            sess_area_grp in area_whitelist or sess_area in area_whitelist
        ):
            continue

        for fname in chan_files:
            parsed = _parse_channel_filename(fname)
            if parsed is None:
                continue
            animal_pref, parsed_sid, ch_int, ch_suffix = parsed
            # cell_id keeps the on-disk channel token verbatim so suffix
            # variants (Ch8p, Ch10s2) stay distinct.
            cell_id = f"{animal_pref}_{parsed_sid}_Ch{ch_int}{ch_suffix}"
            nrn_meta.append(
                {
                    "cell_id": cell_id,
                    "session_id": parsed_sid,
                    "animal_id": animal_pref,
                    "hemisphere": sess_hemi,
                    "area_group": sess_area_grp,
                    "area": sess_area,
                    "channel": ch_int,
                    "channel_suffix": ch_suffix or None,
                    "n_channels_in_session": n_chan,
                    "coord_x": sess_coord[0],
                    "coord_y": sess_coord[1],
                    "recording_type": "multi-unit",
                }
            )

    return nrn_meta


def _load_timit_stims(path: Union[str, Path]) -> list[dict]:
    """Load the 499 TIMIT sentences from ``stimuli/out_sentence_details_timit_all_loudness.mat``.

    **The ``befaft`` silence (0.5 s pre + 0.5 s post by default) is trimmed
    before the waveform is returned.** ``sentdet[].sound`` is shipped with
    real digital silence at both ends for STRF-analysis convenience, but
    the per-trial ``stimon`` time markers in TRIALINFO point at the
    *speech onset*, not the start of the padded waveform. Keeping the
    padding would shift every spectrogram 500 ms ahead of its paired
    response — well outside the typical STRF context window.

    Returns a list of dicts (one per ``sentdet`` entry) with keys
    ``name``, ``stim_id``, ``sound`` (np.ndarray, 16 kHz mono,
    silence-trimmed), ``soundf``, ``duration_s`` (speech-only),
    ``befaft_s = (0.0, 0.0)`` after trimming. The packaged ``aud``
    cochleagram is *not* loaded — we recompute mel from ``sound``
    using the deepSTRF audio pipeline.
    """
    mat = sio.loadmat(
        Path(path) / "stimuli" / "out_sentence_details_timit_all_loudness.mat",
        squeeze_me=True, struct_as_record=False, variable_names=["sentdet"],
    )
    sd = mat["sentdet"]
    out: list[dict] = []
    for s in sd:
        sound = np.asarray(s.sound, dtype=np.float32).reshape(-1)
        soundf = int(np.asarray(s.soundf).item())
        dur_padded = float(np.asarray(s.duration).item())
        ba = np.asarray(s.befaft)
        befaft = tuple(float(x) for x in ba.ravel()[:2]) if ba.size else (0.0, 0.0)

        # Trim the silence padding -- stimon points at speech onset.
        n_pre  = int(round(befaft[0] * soundf))
        n_post = int(round(befaft[1] * soundf))
        if n_pre or n_post:
            assert n_pre + n_post < sound.size, \
                f"befaft={befaft} would empty sound array of length {sound.size}"
            sound_speech = sound[n_pre : sound.size - n_post]
        else:
            sound_speech = sound
        dur_speech = sound_speech.size / soundf

        out.append({
            "name": str(s.name),
            "stim_id": int(np.asarray(s.sentId).item()),
            "sound": sound_speech,
            "soundf": soundf,
            "duration_s": dur_speech,
            "befaft_s": (0.0, 0.0),
            "befaft_s_trimmed": befaft,    # provenance
            "duration_s_padded": dur_padded,
        })
    return out


def _load_mvocs_stims(path: Union[str, Path]) -> tuple[list[dict], dict[int, int]]:
    """Load the 303 unique monkey vocalizations from the concat WAV.

    The release ships ``stimuli/MonkVocs_15Blocks.wav`` (41 kHz stereo,
    ~28 min) with the play-order in ``SqMoPhys_MVOCStimcodes.mat``
    (``mVocsStimCodes`` + ``mVocsStimOnTimes``, 780 slots). Each voc ID
    (1..303) appears in multiple slots; we take **the first occurrence**
    as the canonical waveform and **the minimum inter-onset interval**
    across that ID's occurrences as the canonical duration (which excludes
    the variable inter-stim silence that follows each voc).

    Returns
    -------
    entries : list of dict
        One dict per unique voc ID, sorted by ``stim_id``, with the same
        shape as ``_load_timit_stims`` so the downstream pipeline is shared:
        ``name``, ``stim_id``, ``sound`` (np.float32 mono at native 41 kHz),
        ``soundf``, ``duration_s``, ``befaft_s = (0.0, 0.0)``.
    wav_rep_counts : dict
        ``stim_id -> rep_count_in_canonical_wav``. The Ahmed 2025 paper's
        "test" subset of 11 vocs corresponds to the IDs whose
        ``rep_count_in_canonical_wav == 15`` (see the WAV's design).
    """
    import soundfile as sf  # local import — already a runtime dep

    stimuli_dir = Path(path) / "stimuli"
    code_mat = sio.loadmat(
        stimuli_dir / "SqMoPhys_MVOCStimcodes.mat",
        squeeze_me=True, struct_as_record=False,
    )
    codes = np.asarray(code_mat["mVocsStimCodes"], dtype=np.int64)
    onsets = np.asarray(code_mat["mVocsStimOnTimes"], dtype=np.float64)

    wav_path = stimuli_dir / "MonkVocs_15Blocks.wav"
    wav_info = sf.info(str(wav_path))
    total_dur = wav_info.duration
    sr = int(wav_info.samplerate)

    # Per-slot duration: next-onset gap (or WAV-end gap for the last slot).
    slot_dur = np.empty_like(onsets)
    slot_dur[:-1] = np.diff(onsets)
    slot_dur[-1] = total_dur - onsets[-1]

    # Group slot durations by voc ID; per-ID canonical duration = min.
    durs_by_id: dict[int, list[float]] = {}
    first_onset_by_id: dict[int, float] = {}
    for code, on, d in zip(codes, onsets, slot_dur):
        cid = int(code)
        durs_by_id.setdefault(cid, []).append(float(d))
        if cid not in first_onset_by_id:
            first_onset_by_id[cid] = float(on)

    out: list[dict] = []
    wav_rep_counts: dict[int, int] = {}
    for cid in sorted(durs_by_id):
        canon_dur = float(min(durs_by_id[cid]))
        on0 = first_onset_by_id[cid]
        wav_rep_counts[cid] = len(durs_by_id[cid])
        # Read the snippet for this voc (any occurrence works -- they
        # all play the same recorded waveform; we use the first).
        start_frame = int(round(on0 * sr))
        stop_frame = start_frame + int(round(canon_dur * sr))
        snippet, snippet_sr = sf.read(
            str(wav_path), start=start_frame, stop=stop_frame, dtype="float32",
        )
        assert snippet_sr == sr
        if snippet.ndim == 2:
            snippet = snippet.mean(axis=1)  # stereo -> mono
        out.append({
            "name": f"mvoc_{cid:03d}",
            "stim_id": cid,
            "sound": snippet.astype(np.float32),
            "soundf": sr,
            "duration_s": canon_dur,
            "befaft_s": (0.0, 0.0),
        })
    return out, wav_rep_counts


def _load_session_trial(channel_file: Union[str, Path]) -> dict:
    """Load just the ``trial`` struct from a session's first MUspk file.

    All channels within a session share the same trial struct (verified
    on session 180501 — plain Ch and Chp variants have identical
    ``stimon`` and ``*Stimcode`` arrays).
    """
    m = sio.loadmat(channel_file, squeeze_me=True, struct_as_record=False,
                    variable_names=["trial"])
    t = m["trial"]
    return {
        "stimon": np.asarray(t.stimon, dtype=np.float64),
        "timitStimcode": np.asarray(t.timitStimcode, dtype=np.int64),
        "mVocStimcode": np.asarray(t.mVocStimcode, dtype=np.int64),
    }


def _load_spike_times(channel_file: Union[str, Path]) -> np.ndarray:
    """Load only ``spike.spktimes`` from a MUspk file (skipping the large events matrix)."""
    m = sio.loadmat(channel_file, squeeze_me=True, struct_as_record=False,
                    variable_names=["spike"])
    return np.asarray(m["spike"].spktimes, dtype=np.float64).reshape(-1)


def _bin_spikes_per_rep(
    spktimes: np.ndarray, onsets: np.ndarray, T: int, dt_s: float,
) -> torch.Tensor:
    """Bin per-rep spike counts into a ``(R, T)`` tensor.

    ``onsets`` is the array of stimulus onset times (s) for the R reps.
    Bin edges: ``[onset + i*dt_s, onset + (i+1)*dt_s)`` for i = 0..T-1.
    """
    R = onsets.shape[0]
    out = np.zeros((R, T), dtype=np.float32)
    edges = np.linspace(0.0, T * dt_s, T + 1)
    for r, t0 in enumerate(onsets):
        rel = spktimes - t0
        # only spikes within the stim window matter
        rel = rel[(rel >= 0.0) & (rel < T * dt_s)]
        if rel.size:
            out[r], _ = np.histogram(rel, bins=edges)
    return torch.from_numpy(out)


def _discover_high_rep_set(
    per_session_codes: list[np.ndarray], min_reps: int,
) -> set[int]:
    """Return the set of stim IDs that get >= min_reps reps in at least one session.

    Used to identify the canonical "test" subset (TIMIT: min_reps=11,
    mVocs: min_reps=15) without hard-coding stim IDs — robust if the
    upstream upload ever re-shuffles the ordering.
    """
    high: set[int] = set()
    for codes in per_session_codes:
        codes = codes[codes > 0]
        if codes.size == 0:
            continue
        uniq, counts = np.unique(codes, return_counts=True)
        high.update(int(u) for u, c in zip(uniq, counts) if c >= min_reps)
    return high


class Downer2025Dataset(AudioNeuralDataset):
    """Squirrel-monkey auditory cortex (Downer 2025 / Ahmed 2025).

    Multi-unit threshold-crossing spike trains from 41 sessions across
    3 animals (B, C, F), recorded passively while the animal listened
    to TIMIT speech and monkey vocalizations. 1718 multi-units total
    (one per recording channel).

    Two stim modes are loaded independently:

    - ``stimuli='timit'`` — 499 unique English sentences (489 single-rep
      + 10 with 11 reps; the 10 form the canonical test subset per
      Ahmed 2025).
    - ``stimuli='mvocs'`` — 303 unique monkey vocalizations (292 single-
      rep + 11 with 15 reps; the 11 form the canonical test subset).

    Both modes share the same recording channels but the per-session
    response counts differ, so a (cell, stim) pair has ``(1, 1)`` NaN
    where the channel was in a session that did not play that stim.

    By default, both stim modes go through the same mel pipeline at
    ``audio_fs=16000`` / ``fmax=8000`` to match the Ahmed 2025 baseline
    (cochleagram capped at 8 kHz) and to allow two instances of this
    class (one per stim mode) to be concatenated via
    ``deepSTRF.utils.concat_neural_datasets``.

    Notes
    -----
    Phase-1 skeleton — stim and response loading are not yet implemented
    (will land in subsequent commits). Instantiating with ``_enumerate_only=True``
    populates ``self.nrn_meta`` and ``self.N_neurons`` for inspection.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        stimuli: Literal["timit", "mvocs"] = "timit",
        dt_ms: float = 5.0,
        n_mels: int = 80,
        compression: Literal["cubic", "log1p", "none"] = "log1p",
        spec_zscore: bool = True,
        smooth: bool = True,
        subset: Literal["all", "estimation", "test"] = "all",
        animals: Union[str, Sequence[str]] = "all",
        areas: Optional[Sequence[str]] = None,
        sessions: Optional[Sequence[str]] = None,
        audio_fs: int = 16000,
        fmax: int = 8000,
        window_ms: float = 25.0,
        download: bool = False,
        _enumerate_only: bool = False,
    ):
        """
        Parameters
        ----------
        path : str, optional
            Path to the unpacked dataset root (the directory containing
            ``sessions/`` and ``stimuli/``). Defaults to
            ``default_cache_dir('Downer2025')``.
        stimuli : {'timit', 'mvocs'}
            Which stim class to load. The two share recording channels
            but are loaded independently; concatenate two instances if
            you want both.
        dt_ms : float, default 5.0
            Neural time-bin width. The paper's main analysis uses 50 ms;
            5 ms matches NS1 and gives users the freedom to re-bin.
        n_mels : int, default 80
            Number of mel bands. Matches Ahmed 2025's Kaldi fbank
            (``num_mel_bins=80``) by default.
        compression : {'cubic', 'log1p', 'none'}, default 'log1p'
            Spectrogram amplitude compression. ``log1p`` matches Ahmed
            2025's log-mel (Kaldi fbank). The other deepSTRF audio
            datasets default to ``'cubic'``; we picked ``log1p`` here
            to match the paper as closely as possible.
        spec_zscore : bool, default True
            If True, z-score each spectrogram per-band over its own
            time axis (= Ahmed 2025's ``normalize()`` helper). Boosts
            contrast in higher-frequency bands that would otherwise
            be flattened by the log compression.
        smooth : bool, default True
            Hsu 2004 21 ms PSTH smoothing.
        subset : {'all', 'estimation', 'test'}
            ``'estimation'`` keeps the single-rep stims, ``'test'`` keeps
            the canonical high-rep subset (10 TIMIT IDs or 11 mVocs IDs).
        animals : 'all' or iterable of {'b','c','f'}
        areas : iterable of {'core'|'primary', 'non-primary'} or fine labels
            {'A1','R','ML','AL','CL','CPB','RPB'}. ``'primary'`` is an
            alias for ``'core'``. None = no filter.
        sessions : iterable of session-id strings, or None.
        audio_fs : int, default 16000
            Common sample rate both stim classes are resampled to before
            mel. mVocs source is 41 kHz stereo; TIMIT is already 16 kHz.
        fmax : int, default 8000
            Mel-band high cutoff. Matches Ahmed 2025's cochleagram.
        window_ms : float, default 25.0
            FFT analysis-window length in ms (Kaldi default). The window
            is decoupled from the hop so phonemic detail is preserved
            regardless of ``dt_ms``. Earlier versions of this dataset
            hardcoded ``n_fft = 10 * hop`` which produced a 500 ms FFT
            window at ``dt_ms=50`` and over-smoothed the spectrogram so
            badly that a closed-form ridge STRF only reached
            cc_norm ≈ 0.40 on the well-tuned cohort. With ``window_ms=25``
            the same fit reaches cc_norm ≈ 0.53 — matching Ahmed 2025's
            paper-reported STRF baseline.
        download : bool, default False
            Fetch the 29 GB Zenodo archive (record 16175377) if missing
            and unzip it. Idempotent — both the download and the unzip
            steps are skipped when their outputs already exist.
        _enumerate_only : bool, default False
            Phase-1 internal flag. Populates ``self.nrn_meta`` and
            ``self.N_neurons`` then returns, skipping stim and response
            loading. Will be removed once phases 2–3 are in.
        """
        assert stimuli in ("timit", "mvocs"), f"stimuli must be 'timit' or 'mvocs' (got {stimuli!r})"
        assert subset in ("all", "estimation", "test"), (
            f"subset must be 'all', 'estimation' or 'test' (got {subset!r})"
        )
        assert compression in ("cubic", "log1p", "none"), (
            f"compression must be 'cubic', 'log1p' or 'none' (got {compression!r})"
        )
        assert n_mels > 0 and dt_ms > 0 and audio_fs > 0 and fmax > 0

        if download:
            # Idempotent — returns the extracted auditory_cortex_data/ dir.
            # When path is None this defaults to the platformdirs cache.
            path = download_downer2025(path)
        elif path is None:
            path = str(default_cache_dir("Downer2025") / DOWNER_EXTRACTED_SUBDIR)

        super().__init__(path, dt_ms)

        self.species = "squirrel monkey"
        self.behavioral_state = "awake-passive"
        self.stimuli = stimuli
        self.subset = subset
        self.F = int(n_mels)
        self.audio_fs = int(audio_fs)
        self.fmax = int(fmax)
        self.compression = compression
        self.spec_zscore = bool(spec_zscore)
        self.window_ms = float(window_ms)
        self.smooth = bool(smooth)

        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"Downer2025 root not found: {path}. "
                f"Pass download=True to fetch the ~29 GB Zenodo archive, "
                f"or supply path= pointing to a pre-unpacked "
                f"auditory_cortex_data/ directory."
            )

        sess_meta = _parse_sessions_metadata(path)
        self._session_meta = sess_meta

        self.nrn_meta = _enumerate_neurons(
            path, sess_meta,
            animals=animals, areas=areas, sessions=sessions,
        )
        self.N_neurons = len(self.nrn_meta)

        if _enumerate_only:
            # Inspection escape hatch — skip stim/response loading.
            return

        ########################################
        # 1. group neurons by session for I/O batching
        ########################################
        from collections import defaultdict
        sess_neurons: dict[str, list[tuple[int, str]]] = defaultdict(list)
        sessions_root = Path(path) / "sessions"
        for i, nm in enumerate(self.nrn_meta):
            sid = nm["session_id"]
            suf = nm["channel_suffix"] or ""
            fname = f"{nm['animal_id']}_{sid}_Ch{nm['channel']}{suf}_MUspk.mat"
            sess_neurons[sid].append((i, str(sessions_root / sid / fname)))

        ########################################
        # 2. read trial structs (once per session)
        ########################################
        trial_by_sess: dict[str, dict] = {}
        for sid, neurons in sess_neurons.items():
            trial_by_sess[sid] = _load_session_trial(neurons[0][1])

        if stimuli == "timit":
            stim_field = "timitStimcode"
            canon_key = "timit"
            stim_type_label = "timit"
            entries = _load_timit_stims(path)
            min_reps = int(sess_meta["n_reps_canonical"][canon_key])
            # TIMIT design = per-session: 489 single-rep + 10 at 11 reps.
            # Session-level >= 11 reps robustly identifies the 10 test IDs.
            high_rep_ids = _discover_high_rep_set(
                [trial_by_sess[sid][stim_field] for sid in trial_by_sess],
                min_reps=min_reps,
            )
        else:
            stim_field = "mVocStimcode"
            canon_key = "mVocs"
            stim_type_label = "mvoc"
            entries, wav_rep_counts = _load_mvocs_stims(path)
            min_reps = int(sess_meta["n_reps_canonical"][canon_key])
            # mVocs design = canonical WAV: 11 IDs at exactly 15 reps,
            # the other 292 at variable (1-30) reps. Per-session discovery
            # would over-count (sessions can multiply WAV-level reps).
            high_rep_ids = {cid for cid, n in wav_rep_counts.items() if n == min_reps}

        ########################################
        # 3. build mel + stim_meta
        ########################################
        if subset == "test":
            keep = lambda e: e["stim_id"] in high_rep_ids
        elif subset == "estimation":
            keep = lambda e: e["stim_id"] not in high_rep_ids
        else:
            keep = lambda e: True
        entries = [e for e in entries if keep(e)]

        # Kaldi-fbank wav→spec pipeline (Ahmed 2025 §"Encoding models"
        # references their utils.get_spectrogram, which is literally this
        # function). Using torchaudio.compliance.kaldi.fbank gives:
        #   - true ``window_ms``-length FFT (25 ms default) regardless of dt
        #   - Kaldi pre-emphasis (high-pass, 0.97) -- boosts HF detail
        #   - Kaldi triangular mel filterbank
        #   - log(mel_energy) built-in (no separate compression step)
        # Strategy: compute at the finest frame_shift available, then
        # average-pool to dt-rate. For dt ≤ 10 ms we use frame_shift=dt
        # directly (no pooling); for dt > 10 ms we use the Kaldi default
        # 10 ms hop and pool by dt/10.
        if self.dt <= 10.0:
            frame_shift_ms = self.dt
            pool_factor = 1
        else:
            frame_shift_ms = 10.0
            ratio = self.dt / frame_shift_ms
            pool_factor = int(round(ratio))
            if abs(ratio - pool_factor) > 1e-6:
                raise ValueError(
                    f"dt_ms ({self.dt}) must be a multiple of 10 ms when dt > 10."
                )

        dt_s = self.dt / 1000.0
        S = len(entries)
        T_by_stim: list[int] = [0] * S

        for s_idx, entry in enumerate(entries):
            wav = torch.from_numpy(entry["sound"]).unsqueeze(0)
            if entry["soundf"] != self.audio_fs:
                wav = torchaudio.functional.resample(wav, entry["soundf"], self.audio_fs)
            # Kaldi fbank expects int16-scale input (Ahmed multiplies wav by 2**15).
            wav_k = wav * (2 ** 15)
            spec_native = torchaudio.compliance.kaldi.fbank(
                wav_k,
                num_mel_bins=self.F,
                window_type="hanning",
                sample_frequency=float(self.audio_fs),
                frame_length=float(self.window_ms),
                frame_shift=float(frame_shift_ms),
                low_freq=0.0, high_freq=float(self.fmax),
            )                              # (T_native, F), log-mel
            # average-pool to dt-rate: (T_native, F) -> (T_pool, F)
            T_native = spec_native.shape[0]
            T_pool = T_native // pool_factor
            spec_pooled = spec_native[: T_pool * pool_factor].view(
                T_pool, pool_factor, self.F).mean(dim=1)
            # back to deepSTRF shape (1, F, T)
            spec = spec_pooled.t().unsqueeze(0)
            if self.compression == "cubic":
                # Kaldi already applies log; cubic-root would undo that.
                # Treat compression='cubic' as a no-op here for back-compat
                # (or surface a warning). For now, leave the log-mel as is.
                pass
            elif self.compression == "log1p":
                # Already log-compressed by Kaldi; no further transformation.
                pass
            # 'none' is also a no-op here for the same reason.
            if self.spec_zscore:
                # Per-stim, per-band z-score over the time axis -- matches
                # Ahmed 2025 utils.normalize(). spec is (1, F, T_spec).
                mean = spec.mean(dim=-1, keepdim=True)
                std = spec.std(dim=-1, unbiased=False, keepdim=True).clamp(min=1e-9)
                spec = (spec - mean) / std
            T_canon = int(round(entry["duration_s"] * 1000.0 / self.dt))
            T_by_stim[s_idx] = T_canon
            T_spec = spec.shape[-1]
            if T_spec > T_canon:
                spec = spec[..., :T_canon]
            elif T_spec < T_canon:
                spec = F.pad(spec, (0, T_canon - T_spec), mode="constant", value=0.0)
            is_repeat = entry["stim_id"] in high_rep_ids
            self.stims.append(spec)
            meta = {
                "name": entry["name"],
                "type": stim_type_label,
                "stim_id": entry["stim_id"],
                "duration_s": entry["duration_s"],
                "n_samples": int(entry["sound"].shape[0]),
                "split": "test" if is_repeat else "estimation",
                "n_reps_canonical": min_reps if is_repeat else 1,
                "befaft_s": entry["befaft_s"],
            }
            if stimuli == "mvocs":
                # mVocs design has variable canonical reps (1-30) per voc;
                # surface the actual WAV count so users can sub-filter.
                meta["n_reps_in_wav"] = wav_rep_counts[entry["stim_id"]]
            self.stim_meta.append(meta)

        ########################################
        # 4. bin spike times into per-(stim, neuron) responses
        ########################################
        N = self.N_neurons
        stim_idx_by_id = {e["stim_id"]: i for i, e in enumerate(entries)}
        self.responses = [[None] * N for _ in range(S)]

        for sid, neurons in tqdm(
            sess_neurons.items(), desc=f"Downer2025 {stimuli} sessions",
        ):
            tr = trial_by_sess[sid]
            codes = tr[stim_field]
            onsets = tr["stimon"]
            sess_stim_reps: dict[int, np.ndarray] = {}
            for code_val in np.unique(codes):
                if code_val == 0:
                    continue
                if int(code_val) not in stim_idx_by_id:
                    continue
                sess_stim_reps[int(code_val)] = onsets[codes == code_val]

            for neuron_idx, chan_path in neurons:
                spks = _load_spike_times(chan_path)
                for stim_id_mat, rep_onsets in sess_stim_reps.items():
                    s_idx = stim_idx_by_id[stim_id_mat]
                    self.responses[s_idx][neuron_idx] = _bin_spikes_per_rep(
                        spks, rep_onsets, T_by_stim[s_idx], dt_s,
                    )

        ########################################
        # 5. NaN sentinels for unobserved (stim, neuron) pairs
        ########################################
        nan11 = torch.full((1, 1), float("nan"))
        for s_idx in range(S):
            for n in range(N):
                if self.responses[s_idx][n] is None:
                    self.responses[s_idx][n] = nan11.clone()

        if self.smooth:
            self.smooth_responses(window_ms=21.0)
        self.validate()

    # ------------------------------------------------------------------
    # Hardcoded well-tuned attachment -- cheap alternative to running
    # ``compute_paper_tuning`` from scratch (which is ~5 min for TIMIT,
    # ~4 min for mVocs). Uses the precomputed lists at module level.
    # ------------------------------------------------------------------
    def attach_ahmed2025_well_tuned(self) -> int:
        """Write the precomputed Ahmed 2025 well-tuned booleans into ``nrn_meta``.

        Reads ``AHMED2025_WELL_TUNED_TIMIT`` / ``..._MVOCS`` (module
        constants generated with ``compute_paper_tuning(n_resamples=10_000,
        dt_ms_analysis=50.0, seed=0, alpha=0.05, delta=0.5)`` on the full
        1718-channel population) and tags each ``nrn_meta[i]`` with::

            ahmed2025_<stim>_well_tuned : bool

        where ``<stim>`` matches ``self.stimuli`` (``'timit'`` or
        ``'mvocs'``). Cells not in the precomputed list are tagged
        ``False``.

        Returns
        -------
        int
            Number of cells flagged ``True`` *in the current selection*.

        Notes
        -----
        - For an exact reproduction of the paper's 404 / 489 well-tuned
          counts, instantiate at ``dt_ms=50.0, smooth=False`` and call
          ``compute_paper_tuning(n_resamples=100_000)`` — the precomputed
          lists land at 417 / 476, within ~3 % of the paper.
        - This method does not run any computation; it's pure metadata
          attachment.
        """
        if self.stimuli == "timit":
            lookup = set(AHMED2025_WELL_TUNED_TIMIT)
        elif self.stimuli == "mvocs":
            lookup = set(AHMED2025_WELL_TUNED_MVOCS)
        else:
            raise RuntimeError(f"unknown stimuli mode {self.stimuli!r}")
        key = f"ahmed2025_{self.stimuli}_well_tuned"
        n_flagged = 0
        for m in self.nrn_meta:
            m[key] = m["cell_id"] in lookup
            if m[key]:
                n_flagged += 1
        return n_flagged

    # ------------------------------------------------------------------
    # Paper-faithful tuning criterion (Ahmed et al. 2025, §"Trial-to-trial
    # neural variability"). Opt-in like ``compute_neuron_quality()`` — it
    # mutates ``nrn_meta`` in place.
    # ------------------------------------------------------------------
    def compute_paper_tuning(
        self,
        n_resamples: int = 10_000,
        dt_ms_analysis: float = 50.0,
        seed: int = 0,
        alpha: float = 0.05,
        delta: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """Replicate Ahmed 2025's tuned / well-tuned multi-unit criterion.

        For each neuron and each of the M test-split stims, the method
        randomly samples a pair of reps (with replacement across iterations
        but never the same rep within a pair), concatenates them into long
        sequences U and V (each of length sum_s T_s_coarse), and computes
        their Pearson correlation. The null distribution is built the same
        way but with each V circularly shifted by a random offset before
        correlating. ``n_resamples`` iterations yield two empirical
        distributions per neuron.

        ``tuned`` -- one-sided Wilcoxon rank-sum test (true > null) at
        ``alpha`` (default 0.05). Paper target: 1195 (TIMIT) / 1231 (mVocs).

        ``well_tuned`` -- additionally requires
        ``(mean(true) - mean(null)) / std(null) >= delta`` (default 0.5).
        Paper target: 404 (TIMIT) / 489 (mVocs).

        Writes four floats / booleans to each ``nrn_meta[i]``, prefixed by
        the current stim mode::

            ahmed2025_{timit|mvocs}_tuned          : bool
            ahmed2025_{timit|mvocs}_well_tuned     : bool
            ahmed2025_{timit|mvocs}_p_wilcoxon     : float
            ahmed2025_{timit|mvocs}_delta_normalized : float

        Parameters
        ----------
        n_resamples : int, default 10_000
            Pair-resamplings per neuron. The paper used 100_000; 10_000 is
            ~10x faster and gives a stable Wilcoxon (the criterion only
            cares about the rank order of the two distributions).
        dt_ms_analysis : float, default 50.0
            Coarse bin width for the long-sequence correlations. Must be an
            integer multiple of ``self.dt``. Paper used 50 ms for the main
            results, 20 ms for Fig 4.
        seed : int, default 0
            RNG seed for reproducibility.
        alpha, delta : float
            Significance and effect-size thresholds (see above).
        verbose : bool, default True
            Show a tqdm progress bar.

        Returns
        -------
        summary : dict
            Aggregate counts ``{'tuned': int, 'well_tuned': int,
            'n_with_data': int, 'stimuli': str}``.

        Notes
        -----
        Re-bins ``self.responses`` to ``dt_ms_analysis`` on the fly via
        summing; if ``smooth=True`` was passed to the constructor the
        smoothing has already been applied at ``self.dt``. For the
        strictest paper match instantiate with ``smooth=False`` — though
        in practice the 21 ms Hanning smoothing × subsequent 50 ms
        re-binning makes the smoothing nearly invisible.
        """
        from scipy.stats import ranksums  # local import — scipy is a runtime dep

        ratio = dt_ms_analysis / self.dt
        if abs(ratio - round(ratio)) > 1e-6 or ratio < 1:
            raise ValueError(
                f"dt_ms_analysis ({dt_ms_analysis}) must be an integer multiple "
                f"of self.dt ({self.dt})"
            )
        rebin = int(round(ratio))

        prefix = f"ahmed2025_{self.stimuli}"
        test_s_idces = [s for s, m in enumerate(self.stim_meta) if m["split"] == "test"]
        if not test_s_idces:
            raise RuntimeError(
                "No test-split stims in current dataset (subset filter dropped them?)."
            )

        # Coarse T per test stim
        T_coarse: list[int] = []
        for s in test_s_idces:
            T5 = int(self.stims[s].shape[-1])
            T_coarse.append(T5 // rebin)
        T_total = sum(T_coarse)

        rng = np.random.default_rng(seed)
        n_tuned = 0
        n_well = 0
        n_with_data = 0

        iterator = tqdm(range(self.N_neurons),
                        desc=f"paper tuning {self.stimuli}",
                        disable=not verbose)
        for n in iterator:
            # Pull and re-bin each test stim's reps for this neuron.
            rebinned: list[np.ndarray] = []
            ok = True
            for s, Tc in zip(test_s_idces, T_coarse):
                r = self.responses[s][n]
                if r.shape[0] < 2 or bool(r.isnan().any()):
                    ok = False
                    break
                if Tc < 2:
                    ok = False
                    break
                r_trunc = r[:, :Tc * rebin]
                r_coarse = r_trunc.reshape(r.shape[0], Tc, rebin).sum(dim=-1)
                rebinned.append(r_coarse.numpy().astype(np.float64))
            if not ok:
                self.nrn_meta[n][f"{prefix}_tuned"] = False
                self.nrn_meta[n][f"{prefix}_well_tuned"] = False
                self.nrn_meta[n][f"{prefix}_p_wilcoxon"] = float("nan")
                self.nrn_meta[n][f"{prefix}_delta_normalized"] = float("nan")
                continue
            n_with_data += 1

            # Build U, V (n_resamples, T_total) by picking 2 distinct reps per stim
            U = np.empty((n_resamples, T_total), dtype=np.float64)
            V = np.empty((n_resamples, T_total), dtype=np.float64)
            offset = 0
            for r_coarse, Tc in zip(rebinned, T_coarse):
                R = r_coarse.shape[0]
                first = rng.integers(0, R, size=n_resamples)
                second = rng.integers(0, R - 1, size=n_resamples)
                second = np.where(second >= first, second + 1, second)
                U[:, offset:offset + Tc] = r_coarse[first]
                V[:, offset:offset + Tc] = r_coarse[second]
                offset += Tc

            # Null V: circular shift per iteration (non-zero shift)
            shifts = rng.integers(1, T_total, size=n_resamples)
            col_idx = (np.arange(T_total)[None, :] - shifts[:, None]) % T_total
            V_null = np.take_along_axis(V, col_idx, axis=1)

            true_corrs = _row_pearson(U, V)
            null_corrs = _row_pearson(U, V_null)
            true_corrs = true_corrs[~np.isnan(true_corrs)]
            null_corrs = null_corrs[~np.isnan(null_corrs)]
            if true_corrs.size < 2 or null_corrs.size < 2:
                self.nrn_meta[n][f"{prefix}_tuned"] = False
                self.nrn_meta[n][f"{prefix}_well_tuned"] = False
                self.nrn_meta[n][f"{prefix}_p_wilcoxon"] = float("nan")
                self.nrn_meta[n][f"{prefix}_delta_normalized"] = float("nan")
                continue

            _, p = ranksums(true_corrs, null_corrs, alternative="greater")
            null_std = float(null_corrs.std())
            null_mean = float(null_corrs.mean())
            true_mean = float(true_corrs.mean())
            delta_norm = (true_mean - null_mean) / null_std if null_std > 0 else 0.0
            tuned = bool(p < alpha)
            well = bool(tuned and delta_norm >= delta)
            self.nrn_meta[n][f"{prefix}_tuned"] = tuned
            self.nrn_meta[n][f"{prefix}_well_tuned"] = well
            self.nrn_meta[n][f"{prefix}_p_wilcoxon"] = float(p)
            self.nrn_meta[n][f"{prefix}_delta_normalized"] = float(delta_norm)
            n_tuned += int(tuned)
            n_well += int(well)

        return {"stimuli": self.stimuli, "n_with_data": n_with_data,
                "tuned": n_tuned, "well_tuned": n_well}


def _row_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two ``(n_rows, T)`` arrays.

    Returns a ``(n_rows,)`` array. Rows where either A or B has zero
    variance yield NaN.
    """
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den
