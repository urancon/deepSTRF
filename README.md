# deepSTRF
*A PyTorch-based library and benchmark for fitting sensory neural responses with deep neural network models*

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg) ![Python 3.10](https://img.shields.io/badge/Python-3.10-green)

Work in progress - Mar. 2025

**Contact:** Ulysse Rançon @[urancon](https://github.com/urancon/)

____

## 🧠 Presentation

This repository is associated with the papers "[A general theoretical framework unifying the adaptive, transient and 
sustained properties of ON and OFF auditory neural responses](https://doi.org/10.1371/journal.pcbi.1012288)" and 
"[Temporal recurrence as a general mechanism to explain neural responses in the auditory system](https://www.biorxiv.org/content/early/2025/01/09/2025.01.08.631909)" by Rançon et al.

It contains major codes for result reproduction. In particular, it provides several publicly available datasets that in
convenient **PyTorch** classes, as well as ready-to-deploy computational models and the AdapTrans model of auditory 
ON/OFF responses.

![placeholder.png](docs/_source/img/homepage_illustration.png)

A detailed documentation can be found here: [deepstrf.readthedocs.io](https://deepstrf.readthedocs.io/) (under construction)


## 🏁 Benchmark

To foster improvement of auditory neural encoding models, we report here information about the best-performing models, 
on each dataset. *Feel free to contact us, if you want to claim a spot on the podium of either dataset !*
A ready-to-deploy PyTorch model class will have to be provided to support your claim, and facilitate the work of future 
researchers.


|   **Dataset**   | **Model backbone** | **Rank** |                  **Remarks**                  | **Params / nrn** | **Perfs <br/>(CCraw / CCnorm) [%]** |                                        **Paper (backbone)**                                         | 
|:---------------:|:------------------:|:--------:|:---------------------------------------------:|:----------------:|:-----------------------------------:|:---------------------------------------------------------------------------------------------------:|
|     **NS1**     |      StateNet      |    🥇    |                   GRU, pop                    |      30,465      |             55.6 / 75.1             |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |          
|                 |    Transformer     |    🥈    |                      pop                      |      29,205      |             53.9 / 73.0             |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |          
|                 |       2D-CNN       |    🥉    |                      pop                      |      36,275      |             51.8 / 70.1             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|   **NAT4 A1**   |      StateNet      |    🥇    |                   LSTM, pop                   |      40,271      |             46.6 / 65.1             |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|                 |       2D-CNN       |    🥈    | [AdapTrans](docs_md/README_AdapTrans.md), pop |      XX,XXX      |             46.4 / 64.5             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|                 |    Transformer     |    🥉    |                      pop                      |      28,437      |             46.6 / 64.4             |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|  **NAT4 PEG**   |    Transformer     |    🥇    |                      pop                      |      28,437      |             39.7 / 55.5             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|                 |       2D-CNN       |    🥈    | [AdapTrans](docs_md/README_AdapTrans.md), pop |      XX,XXX      |             39.2 / 55.2             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|                 |      StateNet      |    🥉    |                   LSTM, pop                   |      40,271      |             38.9 / 54.7             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
| **AA1 Field L** |      StateNet      |    🥇    |                   GRU, pop                    |      24,900      |               / 71.0                |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|                 |    Transformer     |    🥈    |                      pop                      |      29,109      |               / 65.5                |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|                 |       2D-CNN       |    🥉    |                      pop                      |      26,915      |               / 65.0                | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|   **AA1 MLd**   |      StateNet      |    🥇    |                  Mamba, pop                   |      32,334      |               / 73.4                |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|                 |       2D-CNN       |    🥈    |                      pop                      |      29,109      |               / 68.9                |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|                 |    Transformer     |    🥉    |                      pop                      |      34,475      |               / 68.3                | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |


**Note:** _Because all three CRCNS AC1 datasets (Wehr, Asari A1, Asari MGB) are single-unit fitting only and can yield 
very different results depending on how their responses are preprocessed (detrending or not, spikes or raw potential, 
etc.), the benchmark on these datasets will be displayed separately._



## ⚡ Installation

This repository assumes you are comfortable with Python environments and conda. To learn more about conda, please visit
https://anaconda.org/.

First create a conda environment and activate it with the following commands:
```shell
conda create --name deepSTRF_env python=3.8
conda activate deepSTRF_env
```

Then, download this repository and get inside:
```shell
git clone https://github.com/urancon/deepSTRF
cd deepSTRF
```

Install Python libraries and dependencies for this code:
```shell
pip3 install -r requirements.txt
```

Download some datasets by following instructions on the [official documentation](https://deepstrf.readthedocs.io/en/latest/).

Now you should be all set up to reproduce our experiments. Have fun !


## 🤖 Train a model

We use [Weights and Biases](https://wandb.ai/) for logging during model training. It is a popular tool among deep learning 
researchers, as it allows to synchronize, plot, and compare metrics for your different runs on a single cloud space, for
free. A downside is that it requires an account; please refer to their website for more information.

The script `main.py` allows you to reproduce major experiments presented in our paper. To train our model with default 
settings, just execute the following command:
```shell
python3 main.py
```

You can also do your own experiment by changing the hyperparameters ! For instance:
```shell
python3 main.py -option1 value1 -option2 value 2
```

To know more about possible options, please do:
```shell
python3 main.py --help
```


## 💡 Contributing

In building and maintaining this repository, our goal is to contribute to harmonize the preprocessing of datasets and 
model training procedures. 

We provide some guidelines on the data formats, tensors, models, etc. The automatic differentiation and GPU parallelization
enabled by the PyTorch deep learning library makes a good basis for the task of neural response fitting.

If you agree with the open-science philosophy and would like to share your data, you can either contribute to this 
repository (we would be glad to help you doing so) or make your own with a similar architecture.




## 📚 References

This work was made possible by the generous publication of several electrophysiology datasets, mainly hosted on the [CRCNS website](https://crcns.org/). If you find them useful
for your research or use them, please do not forget to cite their corresponding article:
* [NS1](docs/_source/md/README_NS1.md) (audio, ferret, extracellular)
* [NAT4](docs/_source/md/README_NAT4.md) (audio, ferret, 2 cortical areas, extracellular)
* [CRCNS AA1](docs/_source/md/README_CRCNS_AA1.md) (audio, zebra finch, 2 areas, extracellular)
* [CRCNS AA2](docs/_source/md/README_CRCNS_AA2.md) (audio, zebra finch, >4 areas, extracellular)
* [CRCNS AC1 - Wehr](docs/_source/md/README_CRCNS_AC1_Wehr.md) (audio, rat, intracellular)
* [CRCNS AC1 - Asari (MGB + A1)](docs/_source/md/README_CRCNS_AC1_Asari.md) (audio, rat, 2 areas, intracellular)
* [Allen Visual Coding - "Ophys"](docs/_source/md/README_Allen_Ophys.md) (vision, mouse, 6 areas, 2-photon)
* [Allen Visual Coding - "Ecephys"](docs/_source/md/README_Allen_Ecephys.md) (vision, mouse, 6 areas, extracellular)


## 📖 Citation 

This code repository is at the core of two of our papers; if you found this repository useful for your research, please consider citing either in your work.

**Published:**
```text
@article{rancon2024pcb,
    doi = {10.1371/journal.pcbi.1012288},
    author = {Rançon, Ulysse and Masquelier, Timothée and Cottereau, Benoit R.},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {A general model unifying the adaptive, transient and sustained properties of ON and OFF auditory neural responses},
    year = {2024},
    month = {08},
    volume = {20},
    url = {https://doi.org/10.1371/journal.pcbi.1012288},
    pages = {1-32},
    number = {8},
}
```


**Preprint:**
```text
@article {rancon2025statenet,
    author = {Rançon, Ulysse and Masquelier, Timothée and Cottereau, Benoit R.},
    title = {Temporal recurrence as a general mechanism to explain neural responses in the auditory system},
    year = {2025},
    doi = {10.1101/2025.01.08.631909},
    publisher = {Cold Spring Harbor Laboratory},
    url = {https://www.biorxiv.org/content/early/2025/01/09/2025.01.08.631909},
    eprint = {https://www.biorxiv.org/content/early/2025/01/09/2025.01.08.631909.full.pdf},
    journal = {bioRxiv},
}
```

## Future improvements

- concatenate datasets (e.g. CRCNS AA1 and AA2)
- fMRI datasets ? (e.g., CRCNS-VIM2)
- behavioral data as additional input, if relevant ? (e.g. eye tracking)
- include artificial stimuli
- multiprocessing / multi-GPU for faster training
- ...