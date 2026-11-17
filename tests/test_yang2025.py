"""Fast tests for Yang2025Dataset on a synthetic slim cache (same format as the prepare step)."""
import json
import math

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from deepSTRF.datasets.audio._yang2025_prepare import CACHE_VERSION
from deepSTRF.datasets.audio.yang2025 import Yang2025Dataset
from deepSTRF.utils import neural_collate

FS = 30000
MARK_T = 9.7 + 0.0123          # B1 unit 0 spike: first piece of B1_w000 starts at 10.0 - 0.3


def _episode(cache, bird, typ, bid, on, off, split):
    t0, t1 = on - 1.0, off + 1.0
    t = np.arange(int(round((t1 - t0) * FS))) / FS + t0
    wav = np.where((t >= on) & (t < off), 0.3 * np.sin(2 * np.pi * 2500 * t), 0.0)
    rel = f"{bird}/audio/{bid}_0.flac"
    (cache / bird / "audio").mkdir(parents=True, exist_ok=True)
    sf.write(str(cache / rel), (wav * 32767).astype(np.int16), FS, subtype="PCM_16")
    return dict(bird=bird, episode_type=typ, bout_id=bid, onset_s=on, offset_s=off, split=split,
                audio_file=rel, audio_t0_s=t0, audio_t1_s=t1)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("yang2025_cache")
    rng = np.random.default_rng(0)
    eps = [_episode(cache, "B1", "warble", "B1_w000", 10.0, 22.0, "train"),
           _episode(cache, "B1", "call", "B1_c000", 30.0, 30.2, "test"),
           _episode(cache, "B2", "warble", "B2_w000", 5.0, 7.0, "train"),
           _episode(cache, "B2", "call", "B2_c000", 20.0, 20.2, "val")]
    units = {}
    for bird, n_units in (("B1", 3), ("B2", 2)):
        times, unit = [], []
        for n in range(n_units):
            st = np.sort(rng.uniform(0, 32, size=2000))
            if bird == "B1" and n == 0:
                st = st[np.abs(st - MARK_T) > 0.006]
                st = np.sort(np.append(st, MARK_T))
            times.append(st)
            unit.append(np.full(len(st), n, np.int32))
        np.savez(cache / bird / "spikes.npz", spike_times=np.concatenate(times), unit=np.concatenate(unit))
        units[bird] = [dict(cluster_id=10 + n, shank=1, max_channel=n, coordinates=[0.0, 20.0 * n],
                            fr=60.0) for n in range(n_units)]
    (cache / "index.json").write_text(json.dumps(dict(
        version=CACHE_VERSION, audio_fs=FS, birds=["B1", "B2"], source="synthetic", episodes=eps, units=units)))
    return cache


@pytest.fixture(scope="module")
def ds(cache):
    return Yang2025Dataset(path=cache)


def test_shapes_and_meta(ds):
    assert ds.N_neurons == 5 and ds.F == 32
    item = ds[0]
    T = item["stims"].shape[-1]
    assert item["stims"].shape == (1, 32, T)
    assert all(r.shape == (1, T) for r in item["responses"] if not torch.isnan(r).all())
    for k in ("bird", "episode_type", "bout_id", "piece_idx", "n_pieces", "t_start_s",
              "vocal_onset_s", "duration_s", "split"):
        assert k in item["stim_meta"]
    for k in ("bird", "cluster_id", "shank", "max_channel", "coordinates", "fr"):
        assert k in ds.nrn_meta[0]


def test_pieces_and_preroll(cache, ds):
    pieces = [m for m in ds.stim_meta if m["bout_id"] == "B1_w000"]
    assert [m["piece_idx"] for m in pieces] == [0, 1, 2] and pieces[0]["n_pieces"] == 3
    for m in pieces:
        s = ds.stim_meta.index(m)
        assert ds.stims[s].shape[-1] == round((4 + 0.3) * 200)
    whole = Yang2025Dataset(path=cache, birds=["B1"], max_episode_s=None)
    assert sum(m["bout_id"] == "B1_w000" for m in whole.stim_meta) == 1


def test_spike_binning(ds):
    s = next(i for i, m in enumerate(ds.stim_meta) if m["bout_id"] == "B1_w000" and m["piece_idx"] == 0)
    assert ds.stim_meta[s]["t_start_s"] == pytest.approx(9.7)
    r = ds.responses[s][0][0]
    # the marked spike sits 12.3 ms into the item -> bin 2 = [9.710, 9.715) s at 5 ms.
    # Background spikes within +-6 ms of it were removed, so bin 2 holds exactly it.
    assert r[2] == 1


def test_other_bird_is_shared_nan(ds):
    b2 = [n for n, m in enumerate(ds.nrn_meta) if m["bird"] == "B2"]
    s1 = [s for s, m in enumerate(ds.stim_meta) if m["bird"] == "B1"]
    entries = [ds.responses[s][n] for s in s1 for n in b2]
    assert all(e is entries[0] for e in entries)
    assert torch.isnan(entries[0]).all()


def test_select_bird_items_only(cache):
    d = Yang2025Dataset(path=cache)
    d.select_pop_by_nrn_attr("bird", "B1")
    n_b1 = sum(m["bird"] == "B1" for m in d.stim_meta)
    assert len(d) == n_b1
    for i in range(len(d)):
        item = d[i]
        assert item["stim_meta"]["bird"] == "B1"
        assert not all(torch.isnan(r).all() for r in item["responses"])


def test_log_spec_finite(ds):
    assert all(torch.isfinite(s).all() for s in ds.stims)


def test_waveform_mode(cache):
    d = Yang2025Dataset(path=cache, return_waveform=True)
    for s, stim in enumerate(d.stims):
        T = next(r for r in d.responses[s] if not torch.isnan(r).all()).shape[-1]
        assert stim.shape == (1, T * 150)


def test_birds_filter_and_unknown_bird(cache):
    d = Yang2025Dataset(path=cache, birds="B2")
    assert {m["bird"] for m in d.nrn_meta} == {"B2"} and d.N_neurons == 2
    with pytest.raises(ValueError):
        Yang2025Dataset(path=cache, birds=["Zz9"])


def test_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Yang2025Dataset(path=tmp_path / "nothing")


def test_decoding_smoke(ds, tmp_path):
    from deepSTRF.models.audio.decoding import WienerFilter
    from deepSTRF.training import Fitter

    loader = DataLoader(ds, batch_size=2, collate_fn=neural_collate)
    model = WienerFilter(ds.N_neurons, ds.F, n_lags=20).fit(loader, alpha=1.0)
    pred = model(next(iter(loader))["responses"])
    assert pred.shape[:3] == (2, 1, ds.F)
    f = Fitter(model, loader, loader, max_epochs=1, ckpt_path=tmp_path / "best.pt",
               log_fn=lambda d: None)
    f.fit()
