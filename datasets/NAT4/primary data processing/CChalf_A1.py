import os

import numpy as np
import torch
from nems0 import epoch
from nems0.recording import load_recording

# CONFIGURATION

file_list = [
    'A1_TNC015a_abf3608afd9b5c83bf8ed17a5ac61040e9c83261.tgz',
    'A1_TNC020a_2f0624a284401ad0e40680435fc57de3873e0a2c.tgz',
    'A1_ARM030a_9b0586bd962b60ab7b02a21771c98695134b8be8.tgz',
    'A1_TNC008a_2eb34ed73bb45f087a72652b5fcea810a9f2e9c9.tgz',
    'A1_ARM029a_aefada7c479857e55fdafe3e00e94df189250651.tgz',
    'A1_TNC021a_89668bbfb0d89473b8efff14b7b6625bb58060dd.tgz',
    'A1_TNC017a_2d5a6c489bf573e2f762eb2e17ca5478271fcc64.tgz',
    'A1_TNC012a_c677f8485dfd121765cd31de50230196a8a4bcc1.tgz',
    'A1_TNC010a_2f50864a77d8ec2f1786ebd8b0161114a9e905b1.tgz',
    'A1_TNC014a_a0ebe478cdce9612658c29d6766859fd66f1a3b6.tgz',
    'A1_TNC016a_13decf694dbf004e611c8e362e23b6fa7852ee4b.tgz',
    'A1_DRX006b_adb77e6b989c6b08bf7fce0c19ab1a94ba124399.tgz',
    'A1_DRX007a_5409c3cb81c2745cec8af5f9a1402ed24ea7ec7d.tgz',
    'A1_DRX008b_240d4f51148b270f6dc099eaccd4f316ce3021f5.tgz',
    'A1_ARM032a_d28371c918efb9917c560f41e51ff15efdf516c5.tgz',
    'A1_CRD017c_07010f7f2781fc649e4f1f90679212e60a7ff3b4.tgz',
    'A1_CRD016d_f9cf97eab58415d6187e5f104e9df6bf06a9fd99.tgz',
    'A1_TNC013a_cc40dccc6e141410c8b0c16e403b763ad368b170.tgz',
    'A1_ARM031a_e73a3420ba4e26d680d9a8adc5bef1c32f6d9617.tgz',
    'A1_ARM033a_8bc7cdda34517574d7973a1b6352d52d873bad7b.tgz',
    'A1_TNC009a_91819235d1188908cee2787e5769d3613fbd756f.tgz',
    'A1_TNC018a_2d5a31aeb27af29f52739c37061da96fcb058e96.tgz'
]

#  LOAD THE DATA
rng = np.random.default_rng()
signals_dir = "file://"
"""cellid = "DRX006b-128-2"
cellids = pd.read_csv("A1_good_777_neurons.csv")
cellids = cellids.values.tolist()"""
recs = []  # Liste pour stocker tous les rec
tot = 0 #total number of neurons

for filebase in file_list:
    datafile = os.path.join(signals_dir, 'data_files_here/A1_single_sites', filebase)
    rec = load_recording(datafile)
    recs.append(rec)
    tot += len(rec['resp'].chans)

index = 0

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


torch.save(ccmax,"ccmax_a1.pt")
