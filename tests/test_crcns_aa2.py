"""CRCNS-AA2 loader: cell filters.

The data tests need the unpacked AA2 archive: ``$AA2_DATA`` (the folder holding
``all_cells/``, ``all_stims/`` and the csv files) or the deepSTRF cache; they skip
otherwise.
"""
from __future__ import annotations

import csv
import os

import pytest
import torch

from deepSTRF.datasets.audio.crcns_aa2 import CRCNSAA2Dataset


def _aa2_path():
    p = os.environ.get("AA2_DATA")
    if p is None:
        from deepSTRF.utils.data_download import default_cache_dir
        p = str(default_cache_dir("AA2"))
    if not os.path.isdir(os.path.join(p, "all_cells")):
        pytest.skip("CRCNS-AA2 data missing (set $AA2_DATA)")
    return p


def test_stim_class_filter_keeps_only_cells_with_that_class():
    # regression: the old filter removed cells while iterating and kept 119 cells
    # that never heard a songrip
    p = _aa2_path()
    ds = CRCNSAA2Dataset(path=p, areas="all", stimuli=("songrip",), dt_ms=10, smooth=False)
    assert all(os.path.isdir(os.path.join(p, "all_cells", m["cell_id"], "songrip")) for m in ds.nrn_meta)


def test_area_labels_follow_cells_under_animal_filter():
    # regression: the animal filter did not filter the area list (18/19 labels wrong)
    p = _aa2_path()
    region = {r[0]: r[1] for r in csv.reader(open(os.path.join(p, "cell_regions.csv")))}
    ds = CRCNSAA2Dataset(path=p, areas=("Field_L", "mld"), stimuli=("conspecific",),
                         animals=["blabla1904", "pupu1203", "oo1220"], dt_ms=10, smooth=False)
    assert len(ds.nrn_meta) > 0
    assert all(m["area"] == region[m["cell_id"]] for m in ds.nrn_meta)
