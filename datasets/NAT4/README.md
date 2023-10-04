# A Natural Sound Dataset: A1 & PEG (NAT4)

**Dataset Source:** [A1 & PEG Dataset](https://doi.org/10.1101/2022.06.10.495698)

**Original Papers:**
- ["Can deep learning provide a generalizable model for dynamic sound encoding in auditory cortex?"](https://doi.org/10.1101/2022.06.10.495698) by Jacob R. Pennington, Stephen V. David.
- ["A convolutional neural network provides a generalizable model of natural sound coding by neural populations in auditory cortex"](https://doi.org/10.1371/journal.pcbi.1011110) by Pennington JR, David SV.

## Dataset Details:

**Description of Stimuli:**
- 20 repetitions of 18 sounds + 1 repetition of 577 sounds.
- Each stimulus is 1.5 seconds in duration.

**Description of Neurons:**
- Total Number of Neurons: 849 for A1, 398 for PEG
- Valid Neurons: 777 for A1, 339 for PEG
- Valid Neurons Criteria: Auditory neurons (see the papers for further details)

**Available Data:**
- *Needs NEMS0 on Python Data Processing.*
- Two recording objects from NEMS0. Each recording contains:
  - Spectrograms of the entire recording.
  - Responses of the entire recording.
  - Names of sounds and names of neurons.

**Information on:**
- Identification of the 18 high-repetition sounds.
- Identification of the 777 valid A1 neurons and 339 valid PEG neurons.

**Processing Needed:**
1. Remove non-valid neurons.
2. Use NEMS0 functions to separate high (val) and low (est) repetition data.
3. Retain only the sounds corresponding to the appropriate data from "est" and "val".
4. Transform data into matrices.


## Setup using our scripts:

- Create an environment with the `NEMS0` library
```shell
conda create -n NEMS0 python=3.7
conda activate NEMS0
git clone https://github.com/lbhb/NEMS0
pip install -e NEMS0
```
- Go to the [official Zenodo dataset repository](https://zenodo.org/record/8044773) and download the data.
- put the data files into the **data/** folder. That is: 
  * unzipped **"A1_Single_Sites"** + **"PEG_Single_Sites"** folders
  * **"A1_NAT4_ozgf.fs100.ch18.tgz"** + **"PEG_NAT4_ozgf.fs100.ch18.tgz"** archives
  * **"A1_pred_correlation.csv"** + **"PEG_pred_correlation.csv"** tables
- Launch `NAT4_preprocessing.py`, which creates 2 files, named **"nat4_a1.pt"** and **"nat4_peg.pt"**. 
The latter two files constitute the final preprocessed dataset files.
- The `NAT4Dataset()` class can be used with the path ti the **data/** folder  containing these two files.


## TODOs

- clean this file (remove uninformative "dataset details")