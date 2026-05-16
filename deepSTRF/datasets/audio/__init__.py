from .audio_dataset import AudioNeuralDataset
from .NS1_DRC_Dataset import NS1_Dataset
from .NAT4_Dataset import NAT4_Dataset
from .Wehr_Dataset import WehrDataset
from .CRCNS_AA1_Dataset import CRCNS_AA1_Dataset
from .CRCNS_AA2_Dataset import CRCNS_AA2_Dataset
from .CRCNS_AA4_Dataset import CRCNS_AA4_Dataset, AA4_ANIMAL_IDS
from .Espejo_Dataset import Espejo_Dataset
from .Alice_EEG_Dataset import Alice_EEG_Dataset, download_alice_eeg

__all__ = ['AudioNeuralDataset',
           'WehrDataset',
           'NS1_Dataset',
           'NAT4_Dataset',
           'CRCNS_AA1_Dataset',
           'CRCNS_AA2_Dataset',
           'CRCNS_AA4_Dataset', 'AA4_ANIMAL_IDS',
           'Espejo_Dataset',
           'Alice_EEG_Dataset', 'download_alice_eeg']
