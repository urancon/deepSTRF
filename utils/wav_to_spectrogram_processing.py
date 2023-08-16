import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.filterbanks import FilterBank
import scipy.io as scio

def load_data(file_path):
    """
    Load data from various file formats.

    Args:
        file_path (str): Path to the input data file.

    Returns:
        numpy.ndarray or torch.Tensor: Loaded data.
    """
    supported_formats = ['.mat', '.pt', '.npy']  # Add supported formats here

    # Get the file extension
    file_extension = file_path.lower().rsplit('.', 1)[-1]

    if file_extension in supported_formats:
        if file_extension == 'mat':
            data = scio.loadmat(file_path)
            return data['spectro'][None, :].astype(np.double)
        elif file_extension == 'pt':
            return torch.load(file_path)
        elif file_extension == 'npy':
            return np.load(file_path)
    else:
        raise ValueError("Unsupported file format")

def processing(input_file_path):
    """
    Perform spectrogram analysis and visualization on the provided input data.

    Args:
        input_file_path (str): Path to the input data file.
    """
    # Initialize the filterbank
    filterbank = FilterBank(filter_type='gammatone', scale='mel',
                            freq_range=(500, 20000), n_filters=34,
                            sampling_rate=44100, filter_length=1500, energy_stride=240, energy_window_length=500)

    # Load and preprocess the input data
    data = load_data(input_file_path)
    if isinstance(data, np.ndarray):
        sig = torch.tensor(data, dtype=torch.float)
    elif isinstance(data, torch.Tensor):
        sig = data
    else:
        raise ValueError("Unsupported data type")

    sig_conv = filterbank(sig)
    spectrogram_from_data = sig_conv.detach().numpy()[0][:]

    # Create subplots for visualization
    fig, axs = plt.subplots(2, 1)

    # Plot the spectrogram obtained from loaded data
    axs[0].imshow(spectrogram_from_data)
    axs[0].set_title("Spectrogram from loaded data")

    # Visualize the time and frequency domain of the filterbank
    filterbank.plot(domain='time', ax=axs[1])
    filterbank.plot(domain='frequency', ax=axs[1])

    # Save plot
    torch.save(spectrogram_from_data, "processed_wav_spectrogram.pt")

    # Display the plots
    plt.show()

if __name__ == "__main__":
    # Specify the path to the input data file
    input_path = 'input_data.xyz'  # Change this to your input file path

    # Call the main function to perform analysis and visualization
    processing(input_path)
