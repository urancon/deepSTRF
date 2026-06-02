import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from deepSTRF.models.scales import mel_to_Hz, Hz_to_mel, Greenwood, inverse_Greenwood


def get_CFs(min_freq, max_freq, n_freqs, scale):
    """Return ``n_freqs`` cochlear/center frequencies (CFs) on a warped scale.

    Parameters
    ----------
    min_freq, max_freq : float
        Frequency range in Hz.
    n_freqs : int
        Number of CFs to return.
    scale : {'mel', 'greenwood'}
        Frequency-axis warping used to space the CFs.

    Returns
    -------
    torch.Tensor
        The ``n_freqs`` center frequencies in Hz.

    Raises
    ------
    NotImplementedError
        If ``scale`` is not ``'mel'`` or ``'greenwood'``.
    """
    if scale == 'mel':
        min_cf = Hz_to_mel(torch.tensor(min_freq))
        max_cf = Hz_to_mel(torch.tensor(max_freq))
        CFs = torch.linspace(min_cf, max_cf, n_freqs)
        CFs = mel_to_Hz(CFs)
    elif scale == 'greenwood':
        min_cf = Greenwood(torch.tensor(min_freq))
        max_cf = Greenwood(torch.tensor(max_freq))
        CFs = torch.linspace(min_cf, max_cf, n_freqs)
        CFs = inverse_Greenwood(CFs)
    else:
        raise NotImplementedError(f"'scale' argument should be 'mel' or 'greenwood', not '{scale}'")
    return CFs


def freq_to_tau(freqs):
    """Map frequencies (Hz) to midbrain-neuron time constants (ms).

    Parameters
    ----------
    freqs : torch.Tensor
        Frequencies in Hz.

    Returns
    -------
    torch.Tensor
        Associated time constants in ms.

    References
    ----------
    Willmore et al. (2016). "Incorporating Midbrain Adaptation to Mean Sound
    Level Improves Models of Auditory Cortical Processing."
    """
    return 500. - 105. * torch.log10(freqs)


def tau_to_a(time_constants, dt: float = 1):
    """Convert physical time constants (ms) to dimensionless ``a`` parameters.

    Parameters
    ----------
    time_constants : torch.Tensor
        Time constants in ms.
    dt : float, default 1
        Time-step width in ms.

    Returns
    -------
    torch.Tensor
        The corresponding ``a = exp(-dt / tau)`` parameters.
    """
    return torch.exp(- dt / time_constants)


def a_to_tau(a, dt: float = 1):
    """Inverse of :func:`tau_to_a`: ``a`` parameter back to a time constant (ms).

    Parameters
    ----------
    a : torch.Tensor
        Dimensionless ``a`` parameters.
    dt : float, default 1
        Time-step width in ms.

    Returns
    -------
    torch.Tensor
        Time constants in ms.
    """
    return - dt / torch.log(a)


class ICAdaptation(nn.Module):
    """
    High-pass exponential filter with frequency-dependent time constants —
    a paper-faithful re-implementation of the inferior-colliculus
    adaptation prefilter described by Willmore et al. (2016).

    Independently filters each frequency band of an input spectrogram
    along the temporal dimension with a parameterized exponential kernel:

        kernel = [...; -Cwa²; -Cwa; -Cw; +1]    with    C = 1/(... + a² + a + 1)

    where the sum of the negative terms equals ``w``. The filter
    effectively computes the difference between the current value of
    the signal in each frequency band and an exponential average of its
    recent past, then applies a half-wave rectification.

    Parameters
    ----------
    init_a_vals : 1D Tensor of length ``F``
        Per-frequency ``a`` parameters (related to the exponential time
        constant: higher ``a`` → longer time constant).
    kernel_size : int, default 2
        Length of the temporal kernel (in frames).

    References
    ----------
    Willmore, Schoppe, King, Schnupp, Harper (2016). "Incorporating
    Midbrain Adaptation to Mean Sound Level Improves Models of
    Auditory Cortical Processing." J. Neurosci. 36(2): 280–289.
    https://doi.org/10.1523/JNEUROSCI.2441-15.2016

    Notes
    -----
    Intentionally non-learnable: the time constants are derived
    analytically from the cochlear frequency map (see
    ``freq_to_tau``) and are paper-faithful. For a learnable
    extension, use :class:`AdapTrans`.

    Input  shape: ``(B, 1, F, T)``.
    Output shape: ``(B, 1, F, T)``.
    """

    out_channels: int = 1

    def __init__(self, init_a_vals, kernel_size: int = 2):
        super().__init__()
        # make sure passed a is one-dimensional
        assert len(init_a_vals.shape) == 1

        # general attributes
        self.F = len(init_a_vals)
        self.K = kernel_size

        # frozen, paper-faithful (Willmore et al. 2016) — registered as
        # buffer so it follows .to(device) but isn't trained.
        self.register_buffer('a', init_a_vals)

    def build_kernels(self):
        """
        Creates a parametrized kernel: a high-pass exponential filter that
        highlights onsets in the signal.
        """
        device = self.a.device

        # normalization constant
        ones = torch.ones(self.K - 1, device=device)
        rng = torch.arange(0, self.K - 1, device=device)

        C = 1 / (torch.outer(self.a, ones) ** rng).sum(dim=1)

        # kernel begins with an exponential whose elements sum to -w, then finishes with +1
        kernel = torch.ones(self.F, 1, self.K, device=device)
        kernel[:, 0, 1:] = -C.unsqueeze(-1) * (torch.outer(self.a, ones) ** rng)
        kernel = torch.flip(kernel, dims=(2,))

        return kernel

    def forward(self, spectro_in):
        """High-pass filter and half-wave rectify each frequency band.

        Parameters
        ----------
        spectro_in : torch.Tensor
            Input spectrogram of shape ``(B, 1, F, T)`` (B=batch, 1 channel,
            F frequency bands, T timesteps).

        Returns
        -------
        torch.Tensor
            High-pass-filtered, full-wave-rectified spectrogram of shape
            ``(B, 1, F, T)``.
        """
        # reshape input spectrogram from single-channel 2D representation to multi-channel 1D
        spectro_in = spectro_in.squeeze(1)                                      # (B, 1, F, T)  --> (B, F, T)

        # build high-pass exponential kernel
        kernel = self.build_kernels()

        # convolve input spectrogram with the kernels
        spectro_in = F.pad(spectro_in, pad=(self.K-1, 0), mode='replicate')     # (B, F, T)     --> (B, F, T+K-1)
        out = F.conv1d(spectro_in, kernel, stride=1, groups=self.F)             # (B, F, T+K-1) --> (B, F, T)

        # full-wave rectification
        out = torch.relu(out)

        # reshape output from a 1D back to 2D representation
        spectro_out = torch.unsqueeze(out, dim=1)                                        # (B, 1, F, T)

        return spectro_out

    def plot_kernels(self, frequency_bin=0):
        kernel = self.build_kernels()
        filter = kernel[frequency_bin, :].squeeze().detach().cpu().numpy()

        plt.figure()
        plt.stem(torch.arange(0, self.K, 1).numpy(), filter, 'r', markerfmt='ro', label='ICAdaptation')
        plt.legend()
        plt.show()


# Backwards-compatibility alias — the original class name was kept long
# enough to be cited in older notebooks. Will be removed in a future release.
Willmore_Adaptation = ICAdaptation


class AdapTrans(nn.Module):
    """
    Adaptive ON/OFF spectrogram prefilter — the learnable extension of the
    inferior-colliculus adaptation prefilter.

    Computes ON and OFF spectrograms through high-pass exponential
    filters with frequency-dependent, learnable time constants. Each
    frequency band is independently filtered along the temporal
    dimension with a parameterized exponential kernel:

        kernel = [...; -Cwa²; -Cwa; -Cw; +1]    with    C = 1/(... + a² + a + 1)

    where the sum of the negative terms equals ``w``. The filter
    computes the difference between the current value of the signal
    in each frequency band and an exponential average of its recent
    past, with separate ``(a, w)`` pairs giving rise to ON and OFF
    polarities.

    Parameters
    ----------
    init_a_vals : 1D Tensor of length ``F``
        Per-frequency ``a`` parameters (related to the time constant).
    init_w_vals : 1D Tensor of length ``F``
        Per-frequency ``w`` parameters (relative weight of the past
        average vs the present sample).
    kernel_size : int, default 2
        Length of the temporal kernel (in frames).
    learnable : bool, default True
        If True, ``a`` and ``w`` are learnable nn.Parameters; if False,
        they are frozen buffers (still follow ``.to(device)``).

    References
    ----------
    Rançon, Masquelier & Cottereau (2024). "A general model unifying
    the adaptive, transient and sustained properties of ON and OFF
    auditory neural responses." PLOS Computational Biology
    20(8):e1012288. https://doi.org/10.1371/journal.pcbi.1012288

    Notes
    -----
    Input  shape: ``(B, 1, F, T)``.
    Output shape: ``(B, 2, F, T)`` — channel 0 is ON, channel 1 is OFF.
    """

    out_channels: int = 2

    def __init__(self, init_a_vals, init_w_vals, kernel_size: int = 2, learnable: bool = True):
        """
        init_a_vals: a 1D vector of 'a' parameters (related to the time constant of the kernel's exponential). The
         higher the 'a', the higher the corresponding time constant of the exponential

        init_w_vals: a 1D vector of 'w' parameters (representing the weight given to the exponential average of the
         signal in its recent past)

        """
        super(AdapTrans, self).__init__()
        # make sure passed a and w are one-dimensional
        assert init_a_vals.shape == init_w_vals.shape
        assert len(init_a_vals.shape) == 1

        # general attributes
        self.F = len(init_a_vals)
        self.K = kernel_size

        # conversion
        init_d_vals = torch.sqrt(1/torch.Tensor(init_a_vals) - 1)
        init_p_vals = torch.sqrt(1/torch.Tensor(init_w_vals) - 1)

        # parameters (one per freq.) — Parameter when learnable, buffer otherwise.
        # Both follow the module's .to(device); raw tensor attributes do not.
        if learnable:
            self.d_on = Parameter(init_d_vals.clone())
            self.d_off = Parameter(init_d_vals.clone())
            self.p = Parameter(init_p_vals.clone())
        else:
            self.register_buffer('d_on', init_d_vals.clone())
            self.register_buffer('d_off', init_d_vals.clone())
            self.register_buffer('p', init_p_vals.clone())

    def build_kernels(self):
        """
        Creates two parametrized kernels:
         - one for the ON response, highlighting onsets in the signal
         - one for the OFF response (offsets), which is the flipped version of the ON kernel

        """
        kernel_ON = self.ON_kernel(self.d_on, self.p)
        kernel_OFF = self.OFF_kernel(self.d_off, self.p)
        return kernel_ON, kernel_OFF

    def ON_kernel(self, d, p):
        """
        Creates the ON kernel
        """
        device = d.device

        # normalization constant
        a = 1 / (1 + d.pow(2))
        ones = torch.ones(self.K - 1, device=device)
        rng = torch.arange(0, self.K - 1, device=device)

        C = 1 / (torch.outer(a, ones) ** rng).sum(dim=1)

        # ON kernel begins with an exponential whose elements sum to -w, then finishes with +1
        kernel_ON = torch.ones(self.F, 1, self.K, device=device)
        w = 1 / (1 + p.pow(2))

        kernel_ON[:, 0, 1:] = (-C * w).unsqueeze(-1) * (torch.outer(a, ones) ** rng)
        kernel_ON = torch.flip(kernel_ON, dims=(2,))

        return kernel_ON

    def OFF_kernel(self, d, p):
        """
        Creates the OFF kernel — the ON kernel flipped about zero, then
        renormalized so its tail equals -w.
        """
        kernel_ON = self.ON_kernel(d, p)
        w = 1 / (1 + p.pow(2))

        # OFF kernel begins with an exponential whose elements sum to +1, then finishes with -w
        kernel_OFF = - kernel_ON / w.unsqueeze(1).unsqueeze(1)
        kernel_OFF[:, 0, -1] = - w

        return kernel_OFF

    def forward(self, spectro_in):
        """Compute the ON and OFF high-pass-filtered spectrograms.

        Parameters
        ----------
        spectro_in : torch.Tensor
            Input spectrogram of shape ``(B, 1, F, T)`` (B=batch, 1 channel,
            F frequency bands, T timesteps).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, 2, F, T)``: channel 0 is the ON response,
            channel 1 is the OFF response (both half-wave rectified).
        """
        # reshape input spectrogram from single-channel 2D representation to multi-channel 1D
        spectro_in = spectro_in.squeeze(1)                                      # (B, 1, F, T)  --> (B, F, T)

        # build ON and OFF high-pass exponential kernels (live on parameter device, follows .to(device))
        kernel_ON, kernel_OFF = self.build_kernels()

        # convolve input spectrogram with the kernels
        spectro_in = nn.functional.pad(spectro_in, pad=(self.K-1, 0), mode='replicate')                              # (B, F, T)     --> (B, F, T+K-1)
        out_ON = nn.functional.conv1d(spectro_in, kernel_ON, stride=1, groups=self.F)                                # (B, F, T+K-1) --> (B, F, T)
        out_OFF = nn.functional.conv1d(spectro_in, kernel_OFF, stride=1, groups=self.F)                              # (B, F, T+K-1) --> (B, F, T)

        # reshape output from a 1D back to 2D representation
        spectro_out = torch.stack([out_ON, out_OFF], dim=1)                              # (B, 2, F, T)

        return torch.relu(spectro_out)

    def get_a(self):
        a_on = 1 / (1 + self.d_on.cpu().detach().pow(2))
        a_off = 1 / (1 + self.d_off.cpu().detach().pow(2))
        return a_on, a_off

    def get_w(self):
        return 1 / (1 + self.p.cpu().detach().pow(2))

    def plot_kernels(self, frequency_bin=0):
        kernel_ON, kernel_OFF = self.build_kernels()
        ON_filter = kernel_ON[frequency_bin, :].squeeze().detach().cpu().numpy()
        OFF_filter = kernel_OFF[frequency_bin, :].squeeze().detach().cpu().numpy()

        plt.figure()
        plt.stem(torch.arange(0, self.K, 1).numpy(), ON_filter, 'r', markerfmt='ro', label='ON')
        plt.stem(torch.arange(0, self.K, 1).numpy(), OFF_filter, 'b', markerfmt='bo', label='OFF')
        plt.legend()
        plt.show()


def make_prefiltering(kind: str, n_frequency_bands: int, dt: float,
                      min_freq: float = 500.0, max_freq: float = 20000.0,
                      scale: str = 'mel', learnable: bool = True,
                      init_w: float = 0.75) -> nn.Module:
    """
    Factory for constructing a prefilter module from compact arguments.

    Convenience wrapper that derives per-frequency ``a`` (and ``w``)
    initial values from the cochlear frequency map, then instantiates
    the requested prefilter class. Equivalent to building the prefilter
    by hand; the factory exists so that user code does not need to
    repeat the ``get_CFs`` / ``freq_to_tau`` / ``tau_to_a`` pipeline.

    Parameters
    ----------
    kind : {'adaptrans', 'icadaptation', 'willmore'}
        Which prefilter to build. ``'willmore'`` is an alias for
        ``'icadaptation'``.
    n_frequency_bands : int
        Number of input frequency bands ``F`` of the spectrogram.
    dt : float
        Time bin width in milliseconds (matches ``dataset.dt_ms``).
    min_freq, max_freq : float
        Frequency range (in Hz) spanned by the cochlear filterbank that
        produced the spectrogram. Defaults: 500 / 20 000 Hz.
    scale : {'mel', 'greenwood'}, default 'mel'
        Frequency-axis scaling used to derive per-band time constants.
    learnable : bool, default True
        Only relevant for ``'adaptrans'``. ``ICAdaptation`` is always
        frozen (paper-faithful).
    init_w : float, default 0.75
        Only relevant for ``'adaptrans'``: initial value of the past-vs-
        present weight ``w``.

    Returns
    -------
    nn.Module
        Configured prefilter instance with an ``out_channels`` attribute.
    """
    cf = get_CFs(min_freq, max_freq, n_frequency_bands, scale)
    tau = freq_to_tau(cf)
    a = tau_to_a(tau, dt=dt)
    K = round(3 * max(tau).item()) + 1

    kind = kind.lower()
    if kind == 'adaptrans':
        w = torch.ones_like(a) * init_w
        return AdapTrans(init_a_vals=a, init_w_vals=w, kernel_size=K, learnable=learnable)
    if kind in ('icadaptation', 'willmore'):
        return ICAdaptation(init_a_vals=a, kernel_size=K)
    raise ValueError(
        f"Unknown prefilter kind {kind!r}. Currently supported: "
        f"'adaptrans', 'icadaptation' (alias 'willmore')."
    )
