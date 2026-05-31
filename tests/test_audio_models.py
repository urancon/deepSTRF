"""
Forward-shape and causality contract tests for the audio model zoo.

Every concrete model in ``deepSTRF.models.audio`` must satisfy:

1. Construction with sensible defaults works.
2. ``forward((B, 1, F, T))`` returns ``(B, N, 1, T)`` — the canonical
   ``(B, N, R=1, T)`` rank documented in ``model_paradigm.md`` §3.
3. Bitwise causality in eval mode: changing future timesteps leaves past
   outputs unchanged. (Train-mode dropout is allowed to wobble; eval is
   the contract surface.)
4. ``validate()`` passes (so the four-slot template's invariants hold).
5. Plays nicely with ``AdapTrans`` (``C_in == 2``) and ``ICAdaptation``
   (``C_in == 1``) prefilters.
"""
import pytest
import torch

from deepSTRF.models.audio import (
    Linear, LinearNonlinear, NetworkReceptiveField, DNet,
    ConvNet2D, Transformer, StateNet,
)
from deepSTRF.models.prefiltering import make_prefiltering


# Reproducible everywhere — many models contain Parameter(torch.rand(...))
# or Conv2d which depend on global RNG.
def _seed():
    torch.manual_seed(0)


# Common shape parameters
B, F, T_in = 2, 34, 50
T_window = 9
N = 5


# ---------------------------------------------------------------------------
# Factories — one per model. Centralized so the parametrize lists below
# stay readable; each factory takes a fresh prefilter (or None) so smoke
# tests can swap prefilters cleanly.
# ---------------------------------------------------------------------------

def _make_linear(prefilter=None):
    return Linear(n_frequency_bands=F, temporal_window_size=T_window,
                  out_neurons=N, prefiltering=prefilter)


def _make_linear_nonlinear(prefilter=None):
    return LinearNonlinear(n_frequency_bands=F, temporal_window_size=T_window,
                           out_neurons=N, prefiltering=prefilter)


def _make_nrf(prefilter=None):
    return NetworkReceptiveField(n_frequency_bands=F, temporal_window_size=T_window,
                                 n_hidden=8, out_neurons=N, prefiltering=prefilter)


def _make_dnet(prefilter=None):
    return DNet(n_frequency_bands=F, temporal_window_size=T_window,
                n_hidden=8, out_neurons=N, prefiltering=prefilter)


def _make_convnet2d(prefilter=None):
    return ConvNet2D(n_frequency_bands=F, kernel_size=(3, 9),
                     c_hidden=8, n_hidden=16, out_neurons=N, prefiltering=prefilter)


def _make_transformer(prefilter=None):
    return Transformer(n_frequency_bands=F,
                       embedding_dim=32, n_heads=2, n_layers=1,
                       out_neurons=N, prefiltering=prefilter)


def _make_statenet(prefilter=None):
    return StateNet(n_frequency_bands=F, kernel_size=7, hidden_channels=4,
                    rnn_type='GRU', out_neurons=N, prefiltering=prefilter)


# All factories, parametrized for the standard tests below.
ALL_FACTORIES = [
    pytest.param(_make_linear,           id='Linear'),
    pytest.param(_make_linear_nonlinear, id='LinearNonlinear'),
    pytest.param(_make_nrf,              id='NetworkReceptiveField'),
    pytest.param(_make_dnet,             id='DNet'),
    pytest.param(_make_convnet2d,        id='ConvNet2D'),
    pytest.param(_make_transformer,      id='Transformer'),
    pytest.param(_make_statenet,         id='StateNet'),
]


# ---------------------------------------------------------------------------
# Forward output rank: (B, N, R=1, T)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_forward_output_rank_no_prefilter(factory):
    _seed()
    m = factory()
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, N, 1, T_in), \
        f"{type(m).__name__}: expected (B={B}, N={N}, R=1, T={T_in}); got {tuple(y.shape)}"


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_forward_output_rank_with_adaptrans(factory):
    """AdapTrans gives C_in=2 — the model must thread it through correctly."""
    _seed()
    pref = make_prefiltering('adaptrans', n_frequency_bands=F, dt=5.0)
    m = factory(prefilter=pref)
    assert m.C_in == 2, f"AdapTrans should give C_in=2; got C_in={m.C_in}"
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, N, 1, T_in), \
        f"{type(m).__name__}+AdapTrans: expected (B={B}, N={N}, R=1, T={T_in}); got {tuple(y.shape)}"


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_forward_output_rank_with_icadaptation(factory):
    """ICAdaptation gives C_in=1 — single-channel paper-faithful prefilter."""
    _seed()
    pref = make_prefiltering('icadaptation', n_frequency_bands=F, dt=5.0)
    m = factory(prefilter=pref)
    assert m.C_in == 1, f"ICAdaptation should give C_in=1; got C_in={m.C_in}"
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, N, 1, T_in)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_bitwise_causality_in_eval_mode(factory):
    """Changing future input timesteps must not change past output timesteps."""
    _seed()
    m = factory()
    m.eval()
    cut = T_in // 2
    x = torch.randn(B, 1, F, T_in)
    x_perturbed = x.clone()
    x_perturbed[..., cut:] = torch.randn_like(x_perturbed[..., cut:])
    with torch.no_grad():
        y = m(x)
        y_perturbed = m(x_perturbed)
    diff = (y - y_perturbed)[..., :cut].abs().max().item()
    # Allow 1e-6 to absorb rounding in long FP chains (S4 hits ~1e-7).
    assert diff < 1e-5, \
        f"{type(m).__name__}: causality violation; max past diff = {diff:.2e}"


def _make_wav_model(kind):
    """Build a model with a non-Identity CausalMel wav2spec slot. Covers the
    base-template path (Linear) and the two custom-forward models (StateNet,
    Transformer) that route their own forward through ``self.wav2spec``."""
    from deepSTRF.models.audio import Linear, StateNet, Transformer
    from deepSTRF.models.wav2spec import CausalMelSpectrogram
    mel = CausalMelSpectrogram(audio_fs=16000, n_mels=F, hop_ms=5.0, win_ms=25.0)
    if kind == 'Linear':
        return Linear(n_frequency_bands=F, temporal_window_size=9, out_neurons=N, wav2spec=mel)
    if kind == 'StateNet':
        return StateNet(n_frequency_bands=F, kernel_size=7, hidden_channels=4,
                        rnn_type='GRU', out_neurons=N, wav2spec=mel)
    if kind == 'Transformer':
        return Transformer(n_frequency_bands=F, embedding_dim=32, n_heads=2,
                           n_layers=1, out_neurons=N, wav2spec=mel)
    raise ValueError(kind)


@pytest.mark.parametrize("reduce", ['sum', 'last', 'peak'])
def test_waveform_gradmap(reduce):
    """waveform_gradmap returns a (T_audio,) gradient for a wav-native model
    and raises for a spectrogram-input (Identity wav2spec) model."""
    _seed()
    from deepSTRF.models.audio import Linear
    from deepSTRF.models.wav2spec import CausalMelSpectrogram

    mel = CausalMelSpectrogram(audio_fs=16000, n_mels=F, hop_ms=5.0, win_ms=25.0)
    m = Linear(n_frequency_bands=F, temporal_window_size=9, out_neurons=N, wav2spec=mel)
    T_audio = 100 * mel.hop
    x = torch.randn(T_audio) * 0.1
    g = m.waveform_gradmap(x, neuron=0, reduce=reduce)
    assert g.shape == (T_audio,)
    assert torch.isfinite(g).all() and g.abs().sum().item() > 0

    # population gradmap (neuron=None) also works
    g_all = m.waveform_gradmap(x, neuron=None, reduce=reduce)
    assert g_all.shape == (T_audio,)

    # spectrogram-input model has no waveform front-end -> raises
    m_spec = Linear(n_frequency_bands=F, temporal_window_size=9, out_neurons=N)
    with pytest.raises(RuntimeError):
        m_spec.waveform_gradmap(torch.randn(F, 50))


@pytest.mark.parametrize("kind", ['Linear', 'StateNet', 'Transformer'])
def test_bitwise_causality_through_wav2spec(kind):
    """End-to-end causality contract for a model with a non-Identity
    ``wav2spec`` slot: changing future audio samples must not perturb past
    output frames. The wav2spec module is exercised in isolation by
    ``tests/test_wav2spec.py``; this is the composition with the model's
    forward (including StateNet/Transformer, whose custom forwards must call
    ``self.wav2spec`` before their own pipeline).
    """
    _seed()
    hop = 80  # 5 ms at 16 kHz
    T_neural = 100
    T_audio = T_neural * hop
    m = _make_wav_model(kind)
    m.eval()

    cut_neural = T_neural // 2
    cut_audio = cut_neural * hop
    x = torch.randn(B, 1, T_audio) * 0.1
    x_perturbed = x.clone()
    x_perturbed[..., cut_audio:] = torch.randn_like(x_perturbed[..., cut_audio:])

    with torch.no_grad():
        y = m(x)
        y_perturbed = m(x_perturbed)
    diff = (y - y_perturbed)[..., :cut_neural].abs().max().item()
    assert diff < 1e-5, (
        f"{kind}+wav2spec: causality violation; max past diff = {diff:.2e}"
    )


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_validate_passes(factory):
    _seed()
    m = factory()
    m.validate()  # must not raise


# ---------------------------------------------------------------------------
# Output rank invariant: variable-T inputs are honored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
@pytest.mark.parametrize("T_test", [25, 50, 100])
def test_output_T_matches_input_T(factory, T_test):
    """Models must accept arbitrary input lengths and preserve T (causal pad)."""
    _seed()
    m = factory()
    x = torch.randn(B, 1, F, T_test)
    y = m(x)
    assert y.shape[-1] == T_test, \
        f"{type(m).__name__}: input T={T_test} produced output T={y.shape[-1]}"


# ---------------------------------------------------------------------------
# Single-neuron edge case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_single_neuron(factory):
    """N=1 should work — degenerate but supported."""
    _seed()
    # Override out_neurons=1; factories default to N=5.
    if factory is _make_linear:
        m = Linear(n_frequency_bands=F, temporal_window_size=T_window, out_neurons=1)
    elif factory is _make_linear_nonlinear:
        m = LinearNonlinear(n_frequency_bands=F, temporal_window_size=T_window, out_neurons=1)
    elif factory is _make_nrf:
        m = NetworkReceptiveField(n_frequency_bands=F, temporal_window_size=T_window,
                                  n_hidden=8, out_neurons=1)
    elif factory is _make_dnet:
        m = DNet(n_frequency_bands=F, temporal_window_size=T_window, n_hidden=8, out_neurons=1)
    elif factory is _make_convnet2d:
        m = ConvNet2D(n_frequency_bands=F, kernel_size=(3, 9), c_hidden=8, n_hidden=16, out_neurons=1)
    elif factory is _make_transformer:
        m = Transformer(n_frequency_bands=F,
                        embedding_dim=32, n_heads=2, n_layers=1, out_neurons=1)
    elif factory is _make_statenet:
        m = StateNet(n_frequency_bands=F, kernel_size=7, hidden_channels=4,
                     rnn_type='GRU', out_neurons=1)
    else:
        pytest.skip(f"no single-neuron path for {factory}")
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, 1, 1, T_in)


# ---------------------------------------------------------------------------
# Slots — every concrete model populates the four canonical slots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_canonical_slots_populated(factory):
    """All four template slots must hold nn.Module instances after __init__."""
    import torch.nn as nn
    _seed()
    m = factory()
    for slot in ('wav2spec', 'prefiltering', 'core', 'readout'):
        assert hasattr(m, slot), f"{type(m).__name__} missing slot {slot!r}"
        attr = getattr(m, slot)
        assert isinstance(attr, nn.Module), \
            f"{type(m).__name__}.{slot} is {type(attr).__name__}, expected nn.Module"


# ---------------------------------------------------------------------------
# Transformer-specific: context_window and length generalization
# ---------------------------------------------------------------------------

def test_transformer_unlimited_context_is_default():
    """Default context is unlimited (None)."""
    _seed()
    m = Transformer(n_frequency_bands=F, embedding_dim=32, n_heads=2, n_layers=1, out_neurons=N)
    assert m.context_window is None


def test_transformer_context_window_constructable():
    """Setting a finite context_window must work and the model still bit-causal."""
    _seed()
    m = Transformer(n_frequency_bands=F, embedding_dim=32, n_heads=2, n_layers=1,
                    out_neurons=N, context_window=5)
    m.eval()
    x = torch.randn(B, 1, F, T_in)
    cut = T_in // 2
    x_perturbed = x.clone()
    x_perturbed[..., cut:] = torch.randn_like(x_perturbed[..., cut:])
    with torch.no_grad():
        diff = (m(x) - m(x_perturbed))[..., :cut].abs().max().item()
    assert diff < 1e-5, \
        f"Transformer(context_window=5): causality violation; max past diff = {diff:.2e}"


def test_transformer_context_window_localizes_dependency():
    """
    Past output at time t should be unaffected by changes to input at any
    time s with s < t - context_window. I.e. the bound itself should hold:
    the model becomes invariant to changes in the deep past.
    """
    _seed()
    L = 30
    window = 5
    m = Transformer(n_frequency_bands=F, embedding_dim=32, n_heads=2, n_layers=1,
                    out_neurons=N, context_window=window)
    m.eval()
    # We perturb only the FIRST few timesteps. Position t > window should
    # be invariant.
    x = torch.randn(B, 1, F, L)
    x_perturbed = x.clone()
    x_perturbed[..., :2] = torch.randn_like(x_perturbed[..., :2])
    with torch.no_grad():
        y, y_perturbed = m(x), m(x_perturbed)
    # Check the segment [window+2, L) — every output here should be insulated
    # from the [:2] perturbation since the deepest reachable input is at
    # t - window. Note: time_patch_size=1 here, so no extra past pull from the
    # patchifier convolution. With time_patch_size=K, extend the safe segment
    # to [window+K+1, L).
    safe_start = window + 2
    diff = (y - y_perturbed)[..., safe_start:].abs().max().item()
    assert diff < 1e-5, \
        f"context_window=5 should insulate output at t>={safe_start} from past changes; got {diff:.2e}"


def test_transformer_time_patch_size_aggregates_history():
    """time_patch_size > 1 should still produce (B, N, 1, L) and stay causal."""
    _seed()
    m = Transformer(n_frequency_bands=F, embedding_dim=32, n_heads=2, n_layers=1,
                    out_neurons=N, time_patch_size=4)
    m.eval()
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, N, 1, T_in)
    cut = T_in // 2
    x_perturbed = x.clone()
    x_perturbed[..., cut:] = torch.randn_like(x_perturbed[..., cut:])
    with torch.no_grad():
        diff = (m(x) - m(x_perturbed))[..., :cut].abs().max().item()
    assert diff < 1e-5


def test_transformer_freq_patch_size_subdivides_F():
    """freq_patch_size < F should subdivide the frequency axis cleanly."""
    _seed()
    # F=34 is divisible by 17 → F_p = 2 frequency tokens per timestep.
    # token_dim = embed_dim * F_p = 32 * 2 = 64; n_heads=4 divides 64.
    m = Transformer(n_frequency_bands=34,
                    freq_patch_size=17, embedding_dim=32,
                    n_heads=4, n_layers=1, out_neurons=N)
    assert m.F_p == 2 and m.token_dim == 64
    x = torch.randn(B, 1, 34, T_in)
    y = m(x)
    assert y.shape == (B, N, 1, T_in)


# ---------------------------------------------------------------------------
# StateNet recurrent / state-space backbones
# ---------------------------------------------------------------------------
# The native torch backbones (GRU/LSTM/RNN) need no extra dependency; LMU and
# S4 are vendored under deepSTRF.models.dependencies; Mamba is the upstream
# ``mambapy`` PyPI package. This block guarantees every advertised backbone
# constructs, produces the canonical (B, N, 1, T) rank, and stays bit-causal
# in eval mode — the regression guard for the SSM-dependency swap.

STATENET_BACKBONES = ['GRU', 'LSTM', 'RNN', 'vanilla', 'LMU', 'Mamba', 'S4']


@pytest.mark.parametrize("rnn_type", STATENET_BACKBONES)
def test_statenet_backbone_forward_and_causal(rnn_type):
    _seed()
    m = StateNet(n_frequency_bands=F, kernel_size=7, hidden_channels=4,
                 rnn_type=rnn_type, out_neurons=N)
    m.eval()
    x = torch.randn(B, 1, F, T_in)
    cut = T_in // 2
    x_perturbed = x.clone()
    x_perturbed[..., cut:] = torch.randn_like(x_perturbed[..., cut:])
    with torch.no_grad():
        y = m(x)
        y_perturbed = m(x_perturbed)
    assert y.shape == (B, N, 1, T_in), \
        f"StateNet(rnn_type={rnn_type!r}): expected {(B, N, 1, T_in)}; got {tuple(y.shape)}"
    assert torch.isfinite(y).all(), f"StateNet(rnn_type={rnn_type!r}): non-finite output"
    diff = (y - y_perturbed)[..., :cut].abs().max().item()
    assert diff < 1e-5, \
        f"StateNet(rnn_type={rnn_type!r}): causality violation; max past diff = {diff:.2e}"


def test_statenet_mamba_backbone_is_mambapy():
    """The Mamba backbone must resolve to the upstream ``mambapy`` package,
    not a vendored copy (the SSM-dependency swap)."""
    _seed()
    m = StateNet(n_frequency_bands=F, kernel_size=7, hidden_channels=4,
                 rnn_type='Mamba', out_neurons=N)
    assert type(m.rnn).__module__.startswith("mambapy"), \
        f"expected mambapy MambaBlock; got {type(m.rnn).__module__}"


def test_statenet_unknown_backbone_raises():
    _seed()
    with pytest.raises(NotImplementedError):
        StateNet(n_frequency_bands=F, kernel_size=7, hidden_channels=4,
                 rnn_type='NoSuchRNN', out_neurons=N)


def test_transformer_freq_patch_size_must_divide_F():
    """Mismatched freq_patch_size should error cleanly."""
    with pytest.raises(ValueError, match="must divide"):
        Transformer(n_frequency_bands=34, freq_patch_size=5,  # 34 % 5 != 0
                    embedding_dim=32, n_heads=2, n_layers=1, out_neurons=N)


# ---------------------------------------------------------------------------
# STRF_gradmap — should produce one (1, F, T) gradient map per neuron and
# work whether the model is on CPU or GPU.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_strf_gradmap_shape(factory):
    _seed()
    m = factory().eval()
    g = m.STRF_gradmap()
    assert g.shape[0] == N
    assert g.shape[1] == 1
    assert g.shape[2] == F
    # T_eff defaults to model.T (>= 1)
    assert g.shape[3] >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_strf_gradmap_on_cuda():
    """Regression: gradmap must allocate the null stim on the model's
    device, not silently fall back to CPU when the model is on GPU."""
    _seed()
    m = _make_convnet2d().eval().to('cuda')
    g = m.STRF_gradmap()
    assert g.is_cuda
    assert g.shape == (N, 1, F, m.T)
