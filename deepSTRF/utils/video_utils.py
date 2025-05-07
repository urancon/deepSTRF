import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_responses(responses_tensor, kernel='gaussian', sigma_time_step=3, kernel_size=9, padding=3):
    """
    Smooth neural responses using a separable 1D Gaussian convolution along the time dimension.

    This function applies temporal smoothing to the input responses tensor by convolving each neuron's 
    response with a Gaussian kernel. The convolution is performed independently for each neuron via a grouped convolution.

    Parameters:
        responses_tensor (Tensor): A tensor of shape (N_neurons, N_seqs, R, T) representing the responses.
        kernel (str): Type of kernel to use. Currently only 'gaussian' is supported.
        sigma_time_step (int): Standard deviation of the Gaussian kernel in time steps.
        kernel_size (int): Size of the convolution kernel.
        padding (int): Amount of zero-padding added to both sides of the time dimension.

    Returns:
        output (Tensor): The smoothed responses tensor of shape (T, N_seqs, R, N_neurons) after processing.
    
    Processing Steps:
        1. Permute and flatten the responses tensor to shape (N_seqs*R, N_neurons, T).
        2. Create a 1D convolution layer with groups equal to N_neurons.
        3. Build a Gaussian kernel and copy its weights into the convolution layer.
        4. Apply convolution and then unflatten/permutate to return to the original ordering.
    """
    N_neurons, N_seqs, R, T = responses_tensor.shape

    responses_tensor = responses_tensor.permute(1, 2, 0, 3).flatten(start_dim=0, end_dim=1) # --> (N_seqs*R, N_neurons, T)

    conv = nn.Conv1d(N_neurons, N_neurons, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=N_neurons)

    # Create the Gaussian kernel weights
    x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    kernel = torch.exp(-0.5 * (x / sigma_time_step) ** 2)
    kernel /= kernel.sum() 
    kernel = kernel.view(1,1,-1) # (1, 1, kernel_size)

    with torch.no_grad():
        conv.weight.copy_(kernel.expand(N_neurons,1,-1)) 
    
    conv.weight.requires_grad = False

    # Apply convolution
    output = conv(responses_tensor) # --> (N_seqs*R, N_neurons, T)
    output = output.unflatten(0, (N_seqs, R)).permute(2, 0, 1, 3) # --> (T, N_seqs, R, N_neurons)

    return output

def compute_dataset_signal_power(responses):
    """
    Computes the signal power for each neuron and stimulus sequence.
    
    Args:
        responses (Tensor): Neural responses with shape (N_neurons, N_seqs, R, T)
            where R is the number of repeats and T is the number of time points.
    
    Returns:
        Tensor: Computed signal power with shape (N_neurons, N_seqs)
    """
    trials = responses.shape[-2]
    sum_of_trials = torch.sum(responses, dim=-2) # (N_neurons, N_seqs, T)
    var_of_trials = torch.var(responses, dim=-1) # (N_neurons, N_seqs, R)
    
    term_1 = torch.var(sum_of_trials, dim=-1) # (N_neurons, N_seqs)
    term_2 = torch.sum(var_of_trials, dim=-1) # (N_neurons, N_seqs)
    diff   = term_1 - term_2
    return diff / (trials * (trials-1)) 

def add_random_noise(video_set, noise_level):
    """
    Adds random Gaussian noise to the video stimuli.
    
    Args:
        video_set (Tensor): The tensor containing video stimuli.
        noise_level (float): The standard deviation multiplier for the random noise.
    
    Returns:
        Tensor: Noisy video stimuli.
    """
    noise = torch.randn_like(video_set) * noise_level
    videos_noisy = video_set + noise
    return videos_noisy