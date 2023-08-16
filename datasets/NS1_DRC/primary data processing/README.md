# A Natural Sound Dataset 1 (NS1)

**Stimuli Source:** [NS1 Dataset](https://osf.io/ayw2p/)

**Original Paper:** ["Network receptive field modeling reveals extensive integration and multi-feature selectivity in auditory cortical neurons"](https://doi.org/10.1371/journal.pcbi.1005113) by Nicol S. Harper, Oliver Schoppe, Ben D. B. Willmore, Zhanfeng F. Cui, Jan W. H. Schnupp, Andrew J. King.

**Papers Using the Dataset:**
- ["A dynamic network model of temporal receptive fields in primary auditory cortex"](https://doi.org/10.1371/journal.pcbi.1006618) by M. Rahman, B. D. B. Willmore, A. J. King, N. S. Harper.
- ["Measuring the performance of neural models"](https://doi.org/10.3389/fncom.2016.00010) by O. Schoppe, N. S. Harper, B. D. B. Willmore, A. J. King, J. W. H. Schnupp.

**Dataset Details:**

**Description of Stimuli:**
- 20 clips of natural sound (speech, ferret vocalizations, other animal vocalizations, and environmental sounds), each 5 seconds in duration.
- Clips were played in random order, and each clip was repeated 20 times.

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

## **To process using our scripts:**

- put the data (MetadateSHEnCneurons.mat + spikesandwav folder + data.pt) in this folder. Launch NS1_DRC_processing.py.
- the processed data file should appear as a pytorch file called "ns1.pt"
- you can use it right away by creating a NS1_DRC_Dataset object with the path to this pytorch file.


# DRC Dataset

**Dataset Source:** DRC Dataset (Same as NS1)

**Stimuli:** The sound stimuli in the DRC dataset consisted of 12 sound clips each of 5s duration, and each clip was presented 5 times. Each clip was of dynamic random chords, which consisted of a sequence of complex tones, each tone being 25 ms long, presented with no gap between the tones. Each complex tone was composed of 34 superposed pure tones, each of whose levels were independently and randomly chosen.

**Experiment:** Same as NS1

**Neurons:** Same as NS1

**Characteristics:** Same as NS1

Since the DRC dataset is the same as the NS1 dataset, the experiment, neurons, and characteristics are identical. Please refer to the information provided for the NS1 dataset for details.

