import os
import h5py

from deepSTRF.datasets.video import VideoNeuralDataset


class CRCNS_MT1_Dataset(VideoNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the CRCNS MT1 Dataset, featuring single-unit responses to complex
     artificial motion stimuli in macaque MT


     ============= STRUCTURE ==============




    """

    def __init__(self, path: str, spat_res:tuple = (112, 112), seq_len:int = None, optim_set: str='train'):
        """
        Initializes the CRCNS_MT1 Dataset.

        Specific units can be selected according to their index.

        Parameters:
            path (str): Path to the 'Allen_OPhys/data/' folder containing files as indicated in our readme
            areas (tuple of str): recording sites of interest, can be 'VISal', 'VISam', 'VISl', 'VISp', 'VISpm' or 'VISrl'
            spat_res (tuple of int): spatial resolution to downsample the stimuli
            optim_set (str): either of 'train', 'valid', 'test' to select the corresponding official set
        """

        super().__init__(path)

        self.stims = []
        self.resps = []
        self.indices = []
        self.nrn_meta = []

        # load the data
        # TODO
        for cellfile in os.listdir(path):
            print(cellfile)
            with h5py.File(os.path.join(path, cellfile)) as f:
                Xidx_hf = f['Xidx_hf'][()]              # (t<T, 10)
                X_hf = f['X_hf'][()]                    # (T, H, W)
                Y_hf = f['Y_hf'][()]                    # (t<T, 1)

                self.stims.append()
                self.resps.append()

        # general attributes
        self.dt = 0.333 # ms
        self.species = 'macaque'
        self.area = 'MT'
        self.N_neurons = self.resps.shape[0]
        self.I = [0]

    def __len__(self):
        """ Returns the number of samples in the dataset. """
        return len(self.indices[self.I])

    def __getitem__(self, stim_index):
        """Retrieves a single sample from the dataset."""
        index = self.indices[stim_index]
        stim = self.stims[index:index+self.seq_len]
        resp = self.resps[:, :, index:index+self.seq_len]
        return stim, resp


if __name__ == "__main__":
    dataset = CRCNS_MT1_Dataset('CRCNS_MT1/data/')
    print('babo')
