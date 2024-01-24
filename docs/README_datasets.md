# Auditory neural response Datasets

One of the main contributions of this repository is the presence of several off-the-shelf electrophysiology datasets, 
compiled from various sources, preprocessed, and represented by convenient PyTorch classes.


---

## Conventions

A major limitation hindering research in the field of auditory neural response fitting, is the great variability of 
preprocessing methods and formats, due to signals of different nature (e.g. patch-clamp, extracellular recordings),
different groups having different habits, different experimental setups, etc. This is why we tried to adopt a single 
"format" for all datasets present in this repository.

All datasets inherit from `torch.utils.data.dataset.Dataset` and can therefore benefit from other PyTorch utilities, 
such as random splitting, shuffling, concatenating, augmentation, and so on.

Because original data differ a lot between sources, the core and attributes of each dataset class might differ a bit, 
but they still share a lot in common and can be used in very similar ways from the exterior.
Namely, the constructor of dataset classes must include one argument to select neurons according to a criterion (mask, 
index, a label, brain area). By default, all neurons are selected and teh constructor will grab the data for all of them. Similarly, the ability to choose stimuli of different nature (natural, pure tone, chord, etc.) 
should be implemented as well. If several signals are available for each neuron, then an option should be made to choose 
between them.

In this framework, each dataset contain different sets of data for each neuron, notably:
* input stimuli (spectrograms) that were presented to this neuron
* the corresponding response, aligned in time
* per-stimulus metrics, like measures of variability (e.g., *TTRC*, *CC_half*) pre-computed once during dataset instanciation,
to save time during fitting

As mentioned above, the set of units to work with is defined according to the user's criterion, in the constructor. 
This number of neurons becomes the attribute `self.N_neurons`. 

An important thing to note is that depending on the dataset, neurons may have different numbers of stimulus/response pairs,
of different durations. Because of this, it can be necessary to fit computational models *unit by unit*, and then average 
results across units. Therefore, all datasets come with a method `select_neuron(index)` that sets the attribute `self.I` 
to the desired `index` (0 by default at instanciation). This attribute can also be a list of multiple desired index, if
you want to train on a *population*, rather than a single isolated unit. You can then iterate over the stimulus/response 
pairs of the currently selected neurons.

Each call of the dataset or associated dataloader will return 4 things: `spectrogram, responses, ccmax, ttrc`.



---

## Dataset Zoo

Currently, 3 response datasets are available, each with a name for easier communication. Each are more described in their
respective README, which gives link to their orginal paper / public repository.
* [NS1](datasets/NS1_DRC/README.md)
* [Wehr](datasets/CRCNS_AC1_Wehr/README.md)
* [NAT4](datasets/NAT4/README.md)