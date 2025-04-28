# Natural Sound Dataset #1 (NS1)

**Dataset Source:** [NS1 Dataset](https://osf.io/ayw2p/)

**Original Paper:** ["Network receptive field modeling reveals extensive integration and multi-feature selectivity in auditory cortical neurons"](https://doi.org/10.1371/journal.pcbi.1005113) by Nicol S. Harper, Oliver Schoppe, Ben D. B. Willmore, Zhanfeng F. Cui, Jan W. H. Schnupp, Andrew J. King.

**Papers Using the Dataset:**
- ["A dynamic network model of temporal receptive fields in primary auditory cortex"](https://doi.org/10.1371/journal.pcbi.1006618) by M. Rahman, B. D. B. Willmore, A. J. King, N. S. Harper.
- ["Measuring the performance of neural models"](https://doi.org/10.3389/fncom.2016.00010) by O. Schoppe, N. S. Harper, B. D. B. Willmore, A. J. King, J. W. H. Schnupp.

## Dataset Details:

**Description of Stimuli:**
- 20 clips of natural sounds (speech, ferret vocalizations, other animal vocalizations, and environmental sounds), and 12 Dynamic Random Chords (DRCs) each 5 seconds in duration.
- Clips were played in random order
- Natural sound clips were repeated 20 times, DRCs 5 times

**Description of Neurons:**
- Total Number of Neurons: 119
- Valid Neurons: 73
- Valid Neurons Criteria: Noise ratio < 40 (The noise ratio is given)
- Neuron Types: 549 single and multi-unit recordings from six adult pigmented ferrets (five female and one male), among which 284 were single units.

**Available data:**
- *Needs MATLAB Data Processing.* More info in the readme file in the dataset.
- Structured neuron array with 119 items. In each element:
  - Sound waveform of each stimulus (y, fs).
  - Spike times of each repeat of each stimulus.
  - Information on the recording, on the noise ratio, etc.

**Processing needed:**
- Transforming the sound waveform into a 34-band spectrogram.
- Choosing 73 neurons out of 119 based on their noise ratio.
- Transforming the spike times of each repeat of each stimulus into PSTHs.
- What's been done is explained in the source paper.

## **Setup using our scripts:**

0. Download the data at [the original dataset repository](https://osf.io/ayw2p/) as well as the **"ns1_drc_spectrograms.pt"** file
1. Put files (**MetadateSHEnCneurons.mat** + **spikesandwav folder** + **ns1_drc_spectrograms.pt**) inside the **data/** folder. 
2. Launch `NS1_DRC_preprocessing.py`.
3. The processed data file should appear as a pytorch file called **ns1_drc_responses.pt**
4. You can use it right away by creating a `NS1_DRC_Dataset(...)` object with the path to the **data/** folder


## TODOs

- include a script to create the **"ns1_drc_spectrograms.pt"** file from the original dataset repository.
- clean this file (remove uninformative "dataset details")