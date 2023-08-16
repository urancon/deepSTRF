import os

import numpy as np
import torch
from nems0 import epoch
from nems0.recording import load_recording

# CONFIGURATION

file_list = [
'PEG_ARM017a_78f8cd0f60b245f8dd137e6b00306c27f0cae9af.tgz',
'PEG_ARM018a_5e04645168a032cf445a5246a5865c1977a2378c.tgz',
'PEG_ARM019a_54195bae38f79360e61248914f2f035ead3c0279.tgz',
'PEG_ARM021b_ee2e2266509d78ceadee0da7906dadeb049c8909.tgz',
'PEG_ARM022b_5e0509b8a4bf6dd7ad890b620f1313941612e49a.tgz',
'PEG_ARM023a_55b620d7fd2958d3201d347d75c6b1be9aa757d9.tgz',
'PEG_ARM024a_b97ebd6d2780914105a1562ce19942dacaf8abb5.tgz',
'PEG_ARM025a_b6cabae20ac9a9d2863fa68569037b29fb1b62ed.tgz',
'PEG_ARM026b_f230bfe4a990c4a8eadcb843bbcb4deb79c837c9.tgz',
'PEG_ARM027a_c36811efdff0d4136be330e77953f6c6e69c7aec.tgz',
'PEG_ARM028b_f2005ed600d22a854579b7735758144c15c56f46.tgz',
]
#'PEG_TNC022a_5da19f65eab678631540a3074cefc14e80dd076c.tgz' enlevé pour avoir les 398
#  LOAD THE DATA
rng = np.random.default_rng()
signals_dir = "file://"
recs = []  # Liste pour stocker tous les rec
tot = 0 #total number of neurons

for filebase in file_list:
    datafile = os.path.join(signals_dir, 'data_files_here/PEG_single_sites', filebase)
    rec = load_recording(datafile)
    recs.append(rec)
    tot += len(rec['resp'].chans)

index=0

cchalves = torch.Tensor(tot, 18)
ccmax = torch.Tensor(tot, 18)
for rec in recs:
    cells = rec['resp'].chans
    resp = rec['resp'].rasterize()
    sounds = epoch.epoch_names_matching(resp.epochs, "^STIM_00")
    for cell in cells:
        single_cell_resp = resp.extract_channels([cell])
        for id, sound in enumerate(sounds):
            var = np.zeros((1, 126))
            for j in range(len(var[0])):
                # extract responses to a single validation stimulus
                r = single_cell_resp.extract_epoch(sound)
                # plot the raster
                raster = r[:, 0, :]
                raster = rng.permuted(raster, axis=0)
                # Here CChalf
                sub_raster1 = raster[:10, :]
                sub_raster2 = raster[10:20, :]

                sub_raster1 = np.mean(sub_raster1, 0)
                sub_raster2 = np.mean(sub_raster2,0)

                cchalf = np.corrcoef(sub_raster1, sub_raster2)
                if np.isnan(cchalf).any():
                    cchalf = np.nan_to_num(cchalf,nan=0.0001)
                var[0, j] = cchalf[0, 1]
            cchalf = np.mean(var)
            cchalves[index, id] = cchalf
            ccmax[index, id] = np.sqrt(2. / (1 + np.sqrt(1. / pow(cchalf,2))))
        index += 1



torch.save(ccmax,"ccmax_peg.pt")
