import torch
import torch.nn as nn
import torch.nn.functional as F

from deepSTRF.datasets.neural_dataset import NeuralDataset
from deepSTRF.utils.video_utils import smooth_responses, compute_dataset_signal_power, add_random_noise


class VideoNeuralDataset(NeuralDataset):
    """
    Neural dataset class for video stimuli

    Stimuli are in the form of (C, H, W, T) tensors
    TODO: description

    """

    def __init__(self, path: str, dt_ms: float):
        super().__init__(path, dt_ms)

    def get_dt(self):
        """ Returns the time step """
        return self.dt

    def normalize_videos(self, video_tensor):
        """Normalize the video stimuli by dividing it by the time dimension."""
        # shape of video_tensor (N_seqs, C, H, W, T) or (N_neurons, N_seqs, C, H, W, T)
        mean = video_tensor.mean(dim=-1, keepdim=True)
        std = video_tensor.std(dim=-1, keepdim=True)
        return (video_tensor - mean) / (std + 1e-8)
    
    def normalize_responses(self, responses_tensor):
        """Normalize the neural responses by dividing it by the time dimension."""
        # shape of responses_tensor (N_neurons, N_seqs, R, T)
        mean = responses_tensor.mean(dim=-1, keepdim=True)
        std = responses_tensor.std(dim=-1, keepdim=True)
        return (responses_tensor - mean) / (std + 1e-8)
    
    def smooth_responses(self, responses_tensor, sigma_time_step=2, kernel_size=11, padding=5):
        """Smooth the neural responses using a Gaussian kernel."""
        return smooth_responses(responses_tensor, sigma_time_step=sigma_time_step, kernel_size=kernel_size, padding=padding)
    
    def change_video_temporal_resolution(self, video, native_dt: int, target_dt: int):
        """
        Changes the temporal resolution of a video by simple subsampling.
        
        Args:
            video (torch.Tensor): Video tensor of shape (C, H, W, T).
            native_dt (int): Native temporal resolution in milliseconds (e.g., 12).
            target_dt (int): Target temporal resolution in milliseconds.

        Returns:
            torch.Tensor: Resampled video tensor.
            int: New time dimension T after resampling.
        """
        ratio = target_dt // native_dt
        assert target_dt % native_dt == 0, "Target temporal resolution must be an integer multiple of native resolution."
        if ratio == 1:
            return video, video.shape[-1]  

        video = video[..., ::ratio]  # Keep one frame every `ratio` frames
        return video, video.shape[-1]
    
    def change_spatial_resolution(self, video_tensor, spatial_resol):
        """Change the spatial resolution of the video stimuli.

        Handles video_tensor of shape (S, C, H, W, T) or (N, S, C, H, W, T).
        """
        if spatial_resol != (self.H, self.W):
            video_dims = video_tensor.dim()

            if video_dims == 5:
                # (S, C, H, W, T)
                S, C, H, W, T = video_tensor.shape
                video_tensor = video_tensor.permute(0, 4, 1, 2, 3)  # (S, T, C, H, W)
                video_tensor = video_tensor.reshape(S * T, C, H, W)

                video_tensor = torch.nn.functional.interpolate(
                    video_tensor,
                    size=spatial_resol,
                    mode='bilinear',
                    antialias=True
                )

                video_tensor = video_tensor.view(S, T, C, *spatial_resol).permute(0, 2, 3, 4, 1)  # (S, C, H, W, T)

            elif video_dims == 6:
                # (N, S, C, H, W, T)
                N, S, C, H, W, T = video_tensor.shape
                video_tensor = video_tensor.permute(0, 1, 5, 2, 3, 4)  # (N, S, T, C, H, W)
                video_tensor = video_tensor.reshape(N * S * T, C, H, W)

                video_tensor = torch.nn.functional.interpolate(
                    video_tensor,
                    size=spatial_resol,
                    mode='bilinear',
                    antialias=True
                )

                video_tensor = video_tensor.view(N, S, T, C, *spatial_resol).permute(0, 1, 3, 4, 5, 2)  # (N, S, C, H, W, T)

            else:
                raise ValueError(f"Unexpected video tensor shape: {video_tensor.shape}")

            # Update internal resolution
            self.H, self.W = spatial_resol

        return video_tensor
            
    def get_responses_signal_power(self, responses_tensor):
        """Compute the signal power for the neural responses."""
        return compute_dataset_signal_power(responses_tensor) # (N_neurons, N_stimuli)
    
    def from_rgb_to_grayscale(self, video_tensor):
        """Convert RGB video stimuli to grayscale."""
        self.C = 1
        r, g, b = video_tensor.unbind(dim=1)
        return (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=1)
    
    def change_temporal_resolution(self, 
                                   native_temporal_resolution_video,
                                   video_tensor, 
                                   native_temporal_resolution_responses,
                                   responses_tensor, 
                                   new_dt):
        """
        Change the temporal resolution of both video stimuli and responses.

        Handles video tensors of shape (S, C, H, W, T) or (N, S, C, H, W, T).

        Args:
            native_temporal_resolution_video (float): Native video time bin size (ms).
            video_tensor (torch.Tensor): Video tensor.
            native_temporal_resolution_responses (float): Native response time bin size (ms).
            responses_tensor (torch.Tensor): Responses tensor of shape (N, S, R, T).
            new_dt (float): New desired time bin size (ms).

        Returns:
            responses_out (torch.Tensor): Tensor with updated time resolution for responses.
            videos_out (torch.Tensor): Tensor with updated time resolution for videos.
            new_T (int): New time dimension length.
        """
        # Process responses
        if new_dt > native_temporal_resolution_responses:
            dt_factor_resps = int(new_dt / native_temporal_resolution_responses)
            responses_out = responses_tensor.reshape(
                self.N_neurons,
                self.S,
                self.R,
                -1, 
                dt_factor_resps
            ).sum(axis=-1)
        else:
            duplication_factor_responses = int(native_temporal_resolution_responses / new_dt)
            responses_out = torch.repeat_interleave(
                responses_tensor, 
                repeats=duplication_factor_responses, 
                dim=-1
            )
            
        # Process videos
        video_dims = video_tensor.dim()

        if video_dims == 5:
            # (S, C, H, W, T)
            leading_shape = video_tensor.shape[:-1]  # (S, C, H, W)
        elif video_dims == 6:
            # (N, S, C, H, W, T)
            leading_shape = video_tensor.shape[:-1]  # (N, S, C, H, W)
        else:
            raise ValueError(f"Unexpected video tensor shape: {video_tensor.shape}")

        if new_dt < native_temporal_resolution_video:
            duplication_factor_video = int(native_temporal_resolution_video / new_dt)
            videos_out = torch.repeat_interleave(
                video_tensor, 
                repeats=duplication_factor_video, 
                dim=-1
            )
        elif new_dt > native_temporal_resolution_video:
            dt_factor_vids = int(new_dt / native_temporal_resolution_video)
            new_shape = (*leading_shape, -1, dt_factor_vids)  # last two dims: (new T, factor)
            videos_out = video_tensor.reshape(new_shape).sum(dim=-1)
        else:
            videos_out = video_tensor

        new_T = videos_out.shape[-1]
        return responses_out, videos_out, new_T
    
    
    def split_into_clips(self, clip_length_frames: int):
        """
        Splits the responses and videos along the time dimension into fixed-length clips.
        If the total number of time bins is not divisible by the clip length,
        the last clip is zero-padded.

        Args:
            clip_length_frames (int): Desired length of each clip, in number of frames/time bins.

        Returns:
            responses_out (torch.Tensor): Shape (N, S*n_clips, R, T_clip)
            videos_out (torch.Tensor): Shape (S*n_clips, C, H, W, T_clip)
        """

        def split_and_pad(tensor, time_dim=-1):
            segments = list(tensor.split(clip_length_frames, dim=time_dim))
            if segments and segments[-1].shape[time_dim] < clip_length_frames:
                pad_size = clip_length_frames - segments[-1].shape[time_dim]
                pad_config = [0, pad_size]
                segments[-1] = F.pad(segments[-1], pad_config, mode='constant', value=0)
            return segments

        # Split responses: (N, S, R, T)
        responses_segments = split_and_pad(self.responses, time_dim=-1)
        responses_stack = torch.stack(responses_segments, dim=0)  # (n_clips, N, S, R, T_clip)
        responses_stack = responses_stack.permute(1, 0, 2, 3, 4)  # (N, n_clips, S, R, T_clip)
        N, n_clips, S, R, T_clip = responses_stack.shape
        responses_out = responses_stack.reshape(N, n_clips * S, R, T_clip)  # (N, S*n_clips, R, T_clip)

        # Split videos: (S, C, H, W, T)
        videos_segments = split_and_pad(self.videos, time_dim=-1)
        videos_stack = torch.stack(videos_segments, dim=0)  # (n_clips, S, C, H, W, T_clip)
        n_clips_v, S_v, C, H, W, T_clip_v = videos_stack.shape
        videos_out = videos_stack.reshape(n_clips_v * S_v, C, H, W, T_clip_v)  # (S*n_clips, C, H, W, T_clip)

        # Update internal state
        self.S = responses_out.shape[1]  # Total number of clips
        self.T = T_clip  # Clip length in frames

        return responses_out, videos_out
    
    def add_noise_to_videos(self, video_tensor, noise_level=0.1):
        """Add random Gaussian noise to the video stimuli."""
        return add_random_noise(video_tensor, noise_level)