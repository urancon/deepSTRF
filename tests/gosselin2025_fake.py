"""Synthetic mini-release with the exact file layout of Zenodo 17271813.

Used by tests/test_gosselin2025.py. Values are planted so tests can check
sound reordering, re-binning, NaN-repeat handling and hair-cell grouping.
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

FS = 192000

# (category cell as in the CSV — blank = filled down, name, intensity, frequency, duration)
ROWS = [
    ("PURE TONES", "PT_50dB_8kHz", "50", "8", "500"),
    ("", "PT_70dB_8kHz", "70", "8", "500"),
    ("RAMPS", "Ramps_down7050_16khz", "70,50", "16", "500"),       # name typo as in the release
    ("AM", "AM_10Hz_WN", "0,70", "", "500"),
    ("CHIRPS VELOCITY", "Chirpup_6kHz_to_12kHz_in_25_ms", "70", "6,12", "25"),
    ("WHITENOISES", "WN_ramps_500_ms_Down_70dB_50dB", "70-50", "", "500"),
    ("DECOMPOSITIONS", "bird_reconstructed", "70", "", "460"),     # wav is 480 ms
    ("", "bird_Chirp_50to25kHz_0.02s_80dB", "80", "50,25", "20"),
    ("COMPLEX SOUNDS", "bird_50dB", "50", "", "460"),
    ("", "bird_70dB", "70", "", "460"),
    ("", "bird_inverted50dB", "50", "", "460"),
    ("", "dassin_50dB", "50", "", "460"),
]
S = len(ROWS)
WAV_MS = {"bird_reconstructed": 480.0}
N_CN, N_IC, N_AC = 3, 2, 3
CC_GROUP_SIZES = (3, 3, 3, 3)
CC_SPONT = (1.0, 5.0, 20.0)          # within-group spontaneous rates (increasing)


def _wavs(rng):
    out = {}

    def t(ms):
        return np.arange(int(round(ms * FS / 1000))) / FS

    out["PT_50dB_8kHz"] = 146 * np.sin(2 * np.pi * 8000 * t(500))
    out["PT_70dB_8kHz"] = 10 * out["PT_50dB_8kHz"]                   # gain copy (x10 = +20 dB)
    bird = 100 * rng.standard_normal(len(t(460)))
    out["bird_50dB"] = bird
    out["bird_70dB"] = 10 * bird                                     # gain copy
    out["bird_inverted50dB"] = bird[::-1]
    for _, name, _, _, dur in ROWS:
        if name not in out:
            out[name] = 300 * rng.standard_normal(len(t(WAV_MS.get(name, float(dur)))))
    return {k: np.clip(np.round(v), -32768, 32767).astype(np.int16) for k, v in out.items()}


def write_fake_release(root: Path) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    idx_cc = list(range(S))
    idx_cnic = [S - 1 - s for s in range(S)]                         # reversed
    idx_ac = [(s + 3) % S for s in range(S)]                         # rolled
    with open(root / "Sound Parameters.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Category", "Sound Name", "Intensity (dB)", "Frequency (kHz)",
                    "Duration (ms)", "Index in CC data", "Index in CN/IC data", "Index in AC data"])
        for s, (c, n, i, fr, d) in enumerate(ROWS):
            w.writerow([c, n, i, fr, d, idx_cc[s], idx_cnic[s], idx_ac[s]])
    wavs = _wavs(rng)
    with zipfile.ZipFile(root / "Soundset.zip", "w") as z:
        for _, name, *_ in ROWS:
            fname = name.replace("16khz", "16kHz")
            p = root / f"_{fname}.wav"
            sf.write(p, wavs[name], FS, subtype="PCM_16")
            z.write(p, f"Soundset/{fname}.wav")
            p.unlink()

    # CN / IC: one spike per (neuron, sound, rep) at t = 200 + 5*s_csv + n (ms index)
    for area, n_units in (("CN", N_CN), ("IC", N_IC)):
        X = np.zeros((n_units, 900, S, 12), np.int16)
        for s in range(S):
            for n in range(n_units):
                X[n, 200 + 5 * s + n, idx_cnic[s], :] = 1
        if area == "CN":
            X[0, :, :, 11] = 0                                       # a faded repeat: zeros, kept
        np.save(root / f"Dataset_{area}.npy", X)
    # AC: value s_csv + 0.01*n in bin 6; reps 10, 11 NaN; (n=2, csv sound 0) rep 9 NaN too;
    # (n=0, csv sound 1) all NaN; (n=0, csv sound 2) rep 0 NaN
    A = np.zeros((N_AC, 24, S, 12))
    for s in range(S):
        for n in range(N_AC):
            A[n, 6, idx_ac[s], :] = s + 0.01 * n
    A[:, :, :, 10:] = np.nan
    A[2, :, idx_ac[0], 9] = np.nan
    A[0, :, idx_ac[1], :] = np.nan                                   # no valid repeat: sentinel
    A[0, :, idx_ac[2], 0] = np.nan                                   # gap first: not a prefix
    np.save(root / "Dataset_AC.npy", A)
    # CC: groups of 3 fibres, spont increasing within group; evoked = spont + 100*(g+1) in 200..699
    fib = []
    for g, size in enumerate(CC_GROUP_SIZES):
        for k in range(size):
            x = np.full((900, S), CC_SPONT[k])
            x[200:700] += 100.0 * (g + 1)
            fib.append(x)
    np.save(root / "Dataset_CC.npy", np.stack(fib))
    for area, sessions in (("CN", [1, 1, 2]), ("IC", [1, 2]), ("AC", [1, 2, 2])):
        with open(root / f"Info_{area}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Neuron Index", "Recording Session"])
            for i, ses in enumerate(sessions):
                w.writerow([i, ses])
    return {"S": S, "idx_cnic": idx_cnic, "idx_ac": idx_ac, "wav_ms": WAV_MS}
