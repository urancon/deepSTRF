import copy
import os.path
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt


class RahmanDataset(Dataset):
    """
     Dataset object to load the dataset used in Rahman et al.
      - Harper et al. (2016), "Network Receptive Field Modeling Reveals Non-linear Characteristics of Auditory Cortical
         Neurons"
      - Rahman et al. (2019), "A dynamic network model of temporal receptive fields in primary auditory cortex"
      - Rahman et al. (2022), "Simple transformations capture auditory input to cortex"

     In these studies, anesthetized ferrets were presented with natural sounds while auditory cortex neurons were being
     recorded.
     Sounds were encoded into a spectrogram representation using a mel filterbank and a neural network model of the
     auditory periphery was trained so that its output neuron activation would match the recorded ones.

     Therefore, the following dataset is composed of:
      - input data  -->     mel spectrogram of the presented sounds                 --> (n_samples, F, T)
      - targets     -->     probability of firing of the recorded/output neuron     --> (n_samples, T)

    """
    def __init__(self, path: str):

        self.root = path  # root of the folder containing the .mat file (downloaded from github)
        self.mat_path = os.path.join(self.root, "test_data_5ms.mat")
        self.sample_rate = 200  # dt = 5 ms

        print("loading dataset from file...")
        matfile = loadmat(self.mat_path)
        self.spectrograms = torch.from_numpy(matfile['X_nfht']).permute(0, 2, 1, 3).float()   # Shape: (n_samples, 1, F, T) = (20, 1, 34, 999)
        self.responses = torch.from_numpy(matfile['y_nt']).float()                            # Shape: (n_samples, T) = (20, 999)

        print("done !")

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectro = self.spectrograms[index]
        response = self.responses[index]
        return spectro, response

    def get_F(self):
        return self.spectrograms.shape[2]

    def get_T(self):
        return self.spectrograms.shape[3]

    def get_preferred_inputs(self, response_threshold, n_timesteps_before_response):
        """
        TODO: determine relationship to STA ?

        :param n_clusters:
        :param response_threshold:
        :param n_timesteps_before_response:
        :return:
        """
        # let's define the shape of the spectrograms we're looking after
        T = n_timesteps_before_response
        F = self.get_F()

        # find all stimuli in the dataset that elicited a response above the pre-defined threshold

        # preferred stimuli
        preferred_stimuli = []
        corresponding_responses = []
        n_stimuli_found = 0
        i = 1

        # run through all examples
        for response, spectrogram in zip(self.responses, self.spectrograms):

            # run through each example and find timesteps where the neuron was highly activated
            landmark_timesteps = torch.where(response >= response_threshold)[0]

            # TODO: maybe filter out timesteps that are too close to the beginning of the sequence, i.e. < n_timesteps_before_response
            # TODO: in the future, keep them, and replace the missing part of the stimulus with uncorrelated gaussian noise
            landmark_timesteps = landmark_timesteps[landmark_timesteps >= n_timesteps_before_response]

            # TODO: maybe filter out timesteps that are too close to each other (find a heuristic for that)
            #  for the moment, if two timesteps are separated by less than n_timesteps_before_response, we only keep the
            #  first one; it could be (n_timesteps_before_response / 2.) as well
            # TODO: rewrite this loop cleaner !
            landmark_timesteps = landmark_timesteps.tolist()
            retained_timesteps = []
            if len(landmark_timesteps) > 0:  # if timesteps have been found
                retained_timesteps = [landmark_timesteps[0]]
                last_t = landmark_timesteps[0]
                for t in landmark_timesteps[1:]:
                    if (t - n_timesteps_before_response) >= last_t:
                        retained_timesteps.append(t)
                        n_stimuli_found += 1
                        preferred_stimuli.append(spectrogram[:, :, t-n_timesteps_before_response:t])
                        corresponding_responses.append(response[t])
                        last_t = t

            print(f"recording #{i}/{len(self)} ==> landmark timesteps: {retained_timesteps}")
            i += 1

        print(f"found {n_stimuli_found} stimuli eliciting a response above {response_threshold}")

        return preferred_stimuli, corresponding_responses

class NS2Dataset(Dataset):
    """Compared to RahmanDataset :
    X (500/650/...) timebins of 10ms"""
    def __init__(self, spectrograms: torch.Tensor, responses: torch.Tensor):
        # self.root = path # NO PATH
        self.spectrograms = spectrograms
        self.responses = responses
        # self.responses = self.responses / self.responses.max() # Normalize responses between 0 and 1

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectrogram = self.spectrograms[index]
        response = self.responses[index]
        return spectrogram, response

    def get_F(self):
        return self.spectrograms.shape[2]

    def get_T(self):
        return self.spectrograms.shape[3]

class NAT4Dataset(Dataset):
    def __init__(self, filepath):
        data = torch.load(filepath)
        self.spectrograms = data['spectrograms']
        self.responses = data['responses']

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectrogram = self.spectrograms[index]
        response = self.responses[index]
        return spectrogram, response

    def get_F(self):
        return self.spectrograms.shape[2]

    def get_T(self):
        return self.spectrograms.shape[3]

class NAT4(Dataset):
    def __init__(self, estimation, validation):
        self.estimation = estimation
        self.validation = validation
        self.neuron_count = len(validation.responses)

    def prepare_nat4(self, choose_dataset="val", neuron_index=0, area="A1"):
        if choose_dataset == "est":
            dataset = copy.deepcopy(self.estimation)
            dset = self.estimation
        else:
            dataset = copy.deepcopy(self.validation)

        dataset.responses = dataset.responses[neuron_index]
        #self.area = area
        return dataset

class NS1Dataset(Dataset):
    """
     Dataset object to load the dataset used in Rahman et al.
      - Harper et al. (2016), "Network Receptive Field Modeling Reveals Non-linear Characteristics of Auditory Cortical
         Neurons"
      - Rahman et al. (2019), "A dynamic network model of temporal receptive fields in primary auditory cortex"
      - Rahman et al. (2022), "Simple transformations capture auditory input to cortex"

     In these studies, anesthetized ferrets were presented with natural sounds while auditory cortex neurons were being
     recorded.
     Sounds were encoded into a spectrogram representation using a mel filterbank and a neural network model of the
     auditory periphery was trained so that its output neuron activation would match the recorded ones.

     Therefore, the following dataset is composed of:
      - input data  -->     mel spectrogram of the presented sounds                 --> (n_samples, F, T)
      - targets     -->     probability of firing of the recorded/output neuron     --> (n_samples, T)

    """
    def __init__(self, data,neuron_index):
        self.spectrograms = data['spectrograms']            # Shape: (n_samples, 1, F, T) = (20, 1, 34, 999)
        self.responses = data['responses'][neuron_index]    # Shape: (n_samples, T) = (20, 999)
        self.ccmax = data['ccmax'][neuron_index]

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectro = self.spectrograms[index]
        response = self.responses[index]
        ccmax = self.ccmax[index]
        return spectro, response,ccmax

    def get_F(self):
        return self.spectrograms.shape[2]

    def get_T(self):
        return self.spectrograms.shape[3]

if __name__ == "__main__":
    dataset = RahmanDataset("Rahman/")
    print(len(dataset))

    '''
    for i in range(len(dataset)):
        spectrogram, response = dataset[i]
        plt.figure()
        plt.subplot(211)
        plt.imshow(spectrogram.squeeze())
        plt.title("Spectrogram of the presented sound")
        plt.subplot(212)
        plt.plot(response)
        plt.title("Response of the observed neuron")
        plt.show()
    '''

    preferred_stims, corresponding_responses = dataset.get_preferred_inputs(response_threshold=1., n_timesteps_before_response=50)

    '''
    for i in range(len(preferred_stims)):
        spectro_stim = preferred_stims[i]
        plt.figure()
        #plt.title(f"Highly excitatory stimulus #{i+1}/{len(preferred_stims)}")
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.imshow(spectro_stim.squeeze())
        #plt.show()
        plt.savefig(f"/home/ulysse/Desktop/PhD2/Research/Ulysse/Results/neural response fitting/Understanding Dataset/spectrotemporal inputs - thresh 1./excitatory_stim_{i}.png", bbox_inches='tight')
        plt.close()
    '''


    """
    from sklearn.cluster import KMeans
    preferred_points = [stim.squeeze().flatten().cpu().numpy() for stim in preferred_stims]
    k = 3
    clusters = KMeans(k)
    clusters.fit(preferred_points, sample_weight=corresponding_responses)
    centroids = clusters.cluster_centers_  # shape: (K, L)
    centroids = torch.from_numpy(centroids)
    c = centroids.unflatten(1, (34, 100))  # (K, 34, 50) = (K, H, W) with H*W=L

    for i in range(len(c)):
        centroid = centroids[i]
        plt.figure()
        plt.title(f"Weighted K-Means centroid #{i+1}/{len(preferred_stims)}")
        plt.imshow(centroid)
        plt.show()
    """

    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    preferred_points = [stim.squeeze().flatten().cpu().numpy() for stim in preferred_stims]

    # returns an array of length n-1; z[i] has the format [idx1, idx2, dist, sample_count];
    # at iteration i-1 the linkage algorithm decided to merge the clusters (here, samples/spectrograms) with
    # indices 14 and 79, which had a distance of dist, forming a new cluster with a total of sample_count_samples
    Z = linkage(preferred_points, 'ward')
    c, coph_dists = cophenet(Z, pdist(preferred_points))  # cophenetic correlation coefficient
    print(c)  # the closer to 1, the better the clustering preserves the original distances

    # plotting the dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8)
    plt.show()

    print("ok")
