# CRCNS AC1 "Wehr" Dataset

**Dataset source**: https://crcns.org/data-sets/ac/ac-1/about

**Original paper**: *["Linearity of Cortical Receptive Fields measured with Natural Sounds"](https://www.jneurosci.org/content/jneuro/24/5/1089.full.pdf),
Machens et al. (2004), J.Neuroscience*

**Citation**:
```text
@misc{asari_wehr_crcns_ac1,
	title = {Auditory cortex and thalamic neuronal responses to various natural and synthetic sounds},
	url = {https://crcns.org/data-sets/ac/ac-1/about},
	doi = {http://dx.doi.org/10.6080/K0KW5CXR},
	language = {en},
	urldate = {2023-08-22},
	author = {Asari, Hiroki andMachens, Christian K. and Wehr, Michael S. and Zador, Anthony M.},
	year = {2009},
	journal = {},
}
```

## Details

**Audio Stimuli**: 
* natural stimuli, i.e. animal vocalizations, environmental sounds, etc.
* 7.5 - 15 s long on average
* covered freqs from 0 to 22 kHz
* 3 to 63 presented sound clips, depending on neurons

**Neural Responses**
* Sprague Dawley rats
* 25 neurons, of which 3 are unresponsive, and 1 has too few data.
* membrane potentials obtained in patch-clamp (current-clamp mode, I=0)
* pharmacologically blocked action potentials


## Benchmark results

| **Model backbone** | **Rank** | **Remarks** | **Params per nrn** | **Perfs <br/>(CCraw / CCnorm) [%]** |                     **Paper (backbone)**                      | 
|:------------------:|:--------:|:-----------:|:------------------:|:-----------------------------------:|:-------------------------------------------------------------:|
|      StateNet      |    🥇    |    Mamba    |       20,966       |             30.6 / 31.2             | [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)    |          
|        DNet        |    🥈    |             |       10,886       |             23.7 / 24.3             | [Rahman et al.](https://doi.org/10.1371/journal.pcbi.1006618) |          
|        NRF         |    🥉    |             |       40,265       |             23.1 / 23.6             | [Harper et al.](https://doi.org/10.1371/journal.pcbi.1005113) |




## Setup

**Requirements**: A CRCNS account to download the dataset, MATLAB to preprocess it.

TODO: finish this section


