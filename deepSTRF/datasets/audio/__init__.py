from .audio_dataset import AudioNeuralDataset
from .ns1_drc import NS1Dataset
from .nat4 import NAT4Dataset
from .wehr import WehrDataset
from .crcns_aa1 import CRCNSAA1Dataset
from .crcns_aa2 import CRCNSAA2Dataset
from .crcns_aa4 import CRCNSAA4Dataset, AA4_ANIMAL_IDS
from .espejo import EspejoDataset
from .alice_eeg import AliceEEGDataset, download_alice_eeg
from .meliza_2025 import Meliza2025Dataset
from .downer2025 import Downer2025Dataset

__all__ = ['AudioNeuralDataset',
           'WehrDataset',
           'NS1Dataset',
           'NAT4Dataset',
           'CRCNSAA1Dataset',
           'CRCNSAA2Dataset',
           'CRCNSAA4Dataset', 'AA4_ANIMAL_IDS',
           'EspejoDataset',
           'AliceEEGDataset', 'download_alice_eeg',
           'Meliza2025Dataset',
           'Downer2025Dataset']
