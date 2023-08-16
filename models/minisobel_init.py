from utils.filterbanks import FilterBank
from models.prefiltering import FrequencyDependentHighPassExponential
import torch

def get_filters(device):
    print(f"\nselected device: {device}\n")

    n_freqs = 34
    freq_range = (500, 20000)
    filter_length = 1501  # 301  # 1501
    energy_window_length = 442  # 80  # 400
    energy_stride = 221  # 40  # 200
    fbank = FilterBank(filter_type='gammatone', scale='mel',
                       freq_range=freq_range, n_filters=n_freqs,
                       sampling_rate=44100, filter_length=filter_length,
                       energy_window_length=energy_window_length, energy_stride=energy_stride)

    for param in fbank.parameters():
        param.requires_grad_(False)
    fbank.eval()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    fbank = fbank.to(device)

    center_freqs = fbank.CFs
    time_constants = FrequencyDependentHighPassExponential.freq_to_tau(center_freqs)
    a_vals = FrequencyDependentHighPassExponential.tau_to_a(time_constants, dt=5)
    w_vals = torch.ones_like(a_vals) * 0.75

    filters = FrequencyDependentHighPassExponential(init_a_vals=a_vals, init_w_vals=w_vals, kernel_size=200, learnable=True)
    #filters.plot_kernels()

    return filters
