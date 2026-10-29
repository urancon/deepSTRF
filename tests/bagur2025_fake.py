"""Synthetic mini-release of Bagur et al. 2025 (Zenodo 13941450) for fast tests.

Same file layout as the real release: ``Code.zip`` (``Code/Utilities/SoundInfo.mat``),
``Sounds.zip`` (``Sounds/<name>.wav``, 192 kHz PCM16), anatomy files, the five
``Data_*.mat`` (2-photon as MATLAB v7.3 / HDF5, so h5py sees them transposed
(R, S, T, N); electrophysiology as v5 (N, T, S, R), baseline-subtracted per
trial like the release), and ``Data_CC.mat`` (15 noisy, baseline-subtracted
repeats of a tonotopic model).

Planted values:
- 2p: independent random values (within the float16 range of the cache), so
  any axis mix-up changes the values read back;
- ephys: integer counts with one spike at bin ``100 + 10 s + n`` in every
  trial, plus sparse background spikes; the file holds counts minus each
  trial's pre-stimulus (bins 0..99) mean;
- CC: fibre f responds to the 70 dB tone at its best frequency (fibres 0-3:
  4 kHz, 4-7: 9 kHz, 8-11: 24 kHz) from bin 100 on.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
from scipy.io import savemat

FS = 192000
# name, StartFreq, StopFreq, MeanFreq (kHz, NaN when undefined), duration (s)
SOUNDS = [
    ("blank", np.nan, np.nan, np.nan, 1.0),
    ("tono60dB_4kHz", 4, 4, 3.94, 0.5),
    ("tono70dB_4kHz", 4, 4, 3.94, 0.5),
    ("tono80dB_4kHz", 4, 4, 3.94, 0.5),
    ("tono70dB_9kHz", 9, 9, 9.75, 0.5),
    ("tono70dB_24kHz", 24, 24, 24.4, 0.5),
    ("Up4+9kHz", 6.5, 6.5, 6.5, 0.5),
    ("Down4+9kHz", 6.5, 6.5, 6.5, 0.5),
    ("Sin7Hz9kHz", 9, 9, 9, 1.0),
    ("Sin3HzWhitenoise", np.nan, np.nan, np.nan, 1.0),
    ("Sin20Hz24kHz", 24, 24, 24, 1.0),
    ("ChirpUp4kHz60dB100ms", 4, 9, 6.46, 0.1),
    ("ChirpUp4kHz60dB500ms", 4, 9, 6.47, 0.5),
    ("ChirpUpclose4to9kHz60dB", 4, 9, 6.47, 0.5),
    ("ChirpUpclose4to9kHz80dB", 4, 9, 6.47, 0.5),
    ("ChirpDownclose9to4kHz60dB", 9, 4, 6.47, 0.5),
]
S = len(SOUNDS)                                                      # 16
N_2P = {"AC": 3, "TH": 2, "IC": 4}
R = {("AC", "2p"): 14, ("TH", "2p"): 15, ("IC", "2p"): 13, ("IC", "ephys"): 14,
     ("TH", "ephys"): 15}
N_EPHYS = {"IC": 3, "TH": 2}
N_FIBRES = 12
T_2P = -0.4683 + 0.0317 * np.arange(47)
T_EPHYS = -0.495 + 0.005 * np.arange(300)
AMP_60 = 10 ** (36.3 / 20) * np.sqrt(2)                              # RMS 36.3 dB re 1 int16 step


def _tone(f_khz, dur, amp):
    t = np.arange(int(round(dur * FS))) / FS
    return amp * np.sin(2 * np.pi * f_khz * 1000 * t)


def _chirp(f0, f1, dur, amp):
    t = np.arange(int(round(dur * FS))) / FS
    k = np.log(f1 / f0) / dur
    return amp * np.sin(2 * np.pi * f0 * 1000 * (np.exp(k * t) - 1) / k)


def _wavs(rng):
    a70 = AMP_60 * 10 ** 0.5
    out = {
        "tono60dB_4kHz": -_tone(4, 0.5, AMP_60 / 10 ** (11 / 20)),     # 11 dB low, sign-inverted
        "tono70dB_4kHz": _tone(4, 0.5, a70),
        "tono80dB_4kHz": _tone(4, 0.5, a70 * 10 ** 0.5),
        "tono70dB_9kHz": _tone(9, 0.5, a70),
        "tono70dB_24kHz": _tone(24, 0.5, a70),
        "Sin7Hz9kHz": _tone(9, 1.0, a70) * (1 + np.sin(2 * np.pi * 7 * np.arange(FS) / FS)) / 2,
        "Sin3HzWhitenoise": rng.standard_normal(FS) * a70 / 2
        * (1 + np.sin(2 * np.pi * 3 * np.arange(FS) / FS)) / 2,
        "Sin20Hz24kHz": _tone(24, 1.0, a70) * (1 + np.sin(2 * np.pi * 20 * np.arange(FS) / FS)) / 2,
        "ChirpUp4kHz60dB100ms": _chirp(4, 9, 0.1, AMP_60),
        "ChirpUpclose4to9kHz60dB": _chirp(4, 9, 0.5, AMP_60),
        "ChirpDownclose9to4kHz60dB": _chirp(9, 4, 0.5, AMP_60),
    }
    out["ChirpUp4kHz60dB500ms"] = out["ChirpUpclose4to9kHz60dB"].copy()   # duplicate under 2 names
    out["ChirpUpclose4to9kHz80dB"] = out["ChirpUpclose4to9kHz60dB"] * 10.0
    ramp = np.linspace(0.1, 1.0, FS // 2)
    up = (_tone(4, 0.5, a70) + _tone(9, 0.5, a70)) / 2 * ramp
    out["Up4+9kHz"], out["Down4+9kHz"] = up, up[::-1].copy()
    return {k: np.round(v).astype(np.int16) for k, v in out.items()}


def write_fake_release(root) -> dict:
    root = Path(root)
    rng = np.random.default_rng(0)
    names = [s[0] for s in SOUNDS]
    # Code.zip with SoundInfo.mat
    tmp = root / "_SoundInfo.mat"
    savemat(tmp, {"StimName": np.array(names, dtype=object),
                  "StartFreq": np.array([s[1] for s in SOUNDS], float),
                  "StopFreq": np.array([s[2] for s in SOUNDS], float),
                  "MeanFreq": np.array([s[3] for s in SOUNDS], float),
                  "SoundDur": np.array([s[4] for s in SOUNDS], float) - 0.0013})
    with zipfile.ZipFile(root / "Code.zip", "w") as z:
        z.write(tmp, "Code/Utilities/SoundInfo.mat")
    tmp.unlink()
    with zipfile.ZipFile(root / "Sounds.zip", "w") as z:
        for name, x in _wavs(rng).items():
            p = root / f"_{name}.wav"
            sf.write(p, x, FS, subtype="PCM_16")
            z.write(p, f"Sounds/{name}.wav")
            p.unlink()
    # 2-photon, MATLAB v7.3: h5py dataset shape (R, S, T, N)
    for area, n in N_2P.items():
        r = R[(area, "2p")]
        val = rng.normal(0, 0.3, (n, S, r, 47)).astype(np.float32)          # (N, S, R, T)
        with h5py.File(root / f"Data_{area}.mat", "w") as h:
            h["Data"] = val.transpose(2, 1, 3, 0)                                  # (R, S, T, N)
            h["t"] = T_2P[:, None]
    # electrophysiology, v5: (N, T, S, R), counts minus each trial's pre-stimulus mean
    counts = {}
    for area, n in N_EPHYS.items():
        r = R[(area, "ephys")]
        c = (rng.random((n, 300, S, r)) < 0.01).astype(np.float32)
        for s in range(S):
            for k in range(n):
                c[k, 100 + 10 * s + k, s, :] += 1
        counts[area] = c
        x = c - c[:, :100].mean(axis=1, keepdims=True)
        savemat(root / ("Data_ICE.mat" if area == "IC" else "Data_THE.mat"),
                {"Data": x.astype(np.float32), "t": T_EPHYS[None]})
    # cochlear model: (F, T, S, 15), noisy repeats, baseline-subtracted per trial
    bf_tone = {0: "tono70dB_4kHz", 1: "tono70dB_9kHz", 2: "tono70dB_24kHz"}
    cc = rng.normal(0, 5, (N_FIBRES, 300, S, 15)) + 20.0
    for f in range(N_FIBRES):
        cc[f, 100:200, names.index(bf_tone[f // 4]), :] += 300.0
    cc -= cc[:, :100].mean(axis=1, keepdims=True)
    savemat(root / "Data_CC.mat", {"Data": cc.astype(np.float32), "t": T_EPHYS[None]})
    # anatomy
    savemat(root / "AnatInfo_AC.mat", {
        "AnatInfo_AC": {"area": np.array([1, 0, 4], np.uint8), "x": np.array([850.0, 900, 1200]),
                        "y": np.array([1150.0, 700, 950]), "z": np.array([200, 300, 450], np.uint16)},
        "RegionNames_AC": np.array(["AI", "DP", "VPAF", "SRAF", "AAF"], dtype=object)})
    savemat(root / "AnatInfo_ICE.mat", {
        "AnatInfo_IC": {"AP": np.array([5.3, 0.0, 5.5]), "ML": np.array([900.0, 0, 1500]),
                        "DV": np.array([800.0, 0, 300]), "Struct": np.array([1, 0, 4], np.uint8)},
        "RegionNames_IC": np.array(["IC-C", "IC-D", "IC-L"], dtype=object)})
    return {"names": names, "S": S, "counts": counts, "n_2p": N_2P, "n_ephys": N_EPHYS}
