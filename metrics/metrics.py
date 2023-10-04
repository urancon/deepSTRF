import itertools
import numpy as np
import torch


########################################################
# BASICS
########################################################

@torch.no_grad()
def statistical_power(y):
    """
    Computes the power of a time series y constituted of T time-bins, R repeats, and B batches.

    See Sahani and Linden (2003), "How Linear are Auditory cortical Responses ?", NIPS

        "power is used in the sense of average squared deviation from the temporal mean"

    :param y: (B, R, T)
    :return: (B, R)
    """
    return ((y - y.mean(dim=-1).unsqueeze(-1)) ** 2).mean(dim=-1)


@torch.no_grad()
def correlation_coefficient(y_pred, y_gt, reduction="mean"):
    """
    Computes the average correlation coefficient between two batches of tensors, whose last dimension is time.
    It is computed independently for each pair in the batch, and then averaged over all but the temporal dimension.

    :param y_pred: (*, T)
    :param y_gt: (*, T)
    :return: (*,) or a scalar
    """
    cov = covariance(y_pred, y_gt)              # (*,)
    var_pred = torch.var(y_pred, dim=-1)        # (*,)
    var_gt = torch.var(y_gt, dim=-1)            # (*,)
    cc = cov / torch.sqrt(var_pred * var_gt)    # (*,)
    if reduction == 'mean':
        cc = torch.mean(cc)  # scalar
    return cc


@torch.no_grad()
def covariance(y_pred, y_gt):
    """
    Computes the average covariance between two batches of tensors, whose last dimension is time.
    It is computed independently for each pair in the batch.

    :param y_pred: (*, T)
    :param y_gt: (*, T)
    :return: (*, ) or a scalar
    """
    T = y_pred.shape[-1]
    y_pred_avg = torch.mean(y_pred, dim=-1)                     # (*,)
    y_gt_avg = torch.mean(y_gt, dim=-1)                         # (*,)
    y_pred_rect = y_pred - y_pred_avg.unsqueeze(-1)             # (*,)
    y_gt_rect = y_gt - y_gt_avg.unsqueeze(-1)                   # (*,)
    cov = torch.sum(y_pred_rect * y_gt_rect, dim=-1) / (T-1)    # (*,)
    return cov


########################################################
# NORMALIZATION FACTORS
########################################################

@torch.no_grad()
def compute_TTRC(responses):
    """
    Computes the Trial-to-Trial Response Correlation (TTRC).
     Ref:
        Pennington and David (2023), "A convolutional neural network provides a generalizable model of natural sound
        coding by neural populations in auditory cortex", PLOS CB

    Note:
        If only 1 trial (i.e. 1 repeat), the TTRC equals 1. It is logical in the sense that the cross-correlation of a
        trial with itself is a perfect 1. However, a TTRC of 1 is the worst case scenario for normalization purposes, as
        dividing by a factor of 1. won't do anything. So in this case, the raw and normalized correlation coefficients
        are equal.

    :param responses:  (B, R, T) = (batches, N_repeats, N_timebins)
    return: (B, )
    """
    (B, N_repeats, T), device = responses.shape, responses.device
    if N_repeats == 1:
        return torch.ones(B).to(device)
    else:
        ttrc = 0.
        for i in range(N_repeats):
            for j in range(N_repeats):
                if i >= j:
                    pass
                else:
                    ttrc += correlation_coefficient(responses[:, i, :], responses[:, j, :], reduction='None')
        ttrc /= ((N_repeats*N_repeats - N_repeats)/2)
        return ttrc


@torch.no_grad()
def compute_CCmax(responses, max_iters=float('inf')):
    """
    Computes the CCmax normalization factor
    Ref:
        Schoppe et al. (2016),  'Measuring the Performance of Neural Models', Frontiers in Neuroscience

    Note:
        If only 1 repeat (i.e. 1 trial), the CCmax equals 1. It is logical in the sense that the cross-correlation of a
        trial with itself is a perfect 1. However, a CCmax of 1 is the worst case scenario for normalization purposes, as
        dividing by a factor of 1. won't do anything. So in this case, the raw and normalized correlation coefficients
        are equal.

    :param responses: (B, R, T) = (Batch, N_repeats, N_timebins)
    :param max_iters: int, the number of combinations of trials to compute the ccmax
    return: a scalar
    """
    (B, N_repeats, T), device = responses.shape, responses.device
    if N_repeats == 1:
        return torch.ones(B).to(device)
    else:
        if N_repeats % 2 == 1:
            N_repeats -= 1
        repeat_indices = list(range(N_repeats))
        half_sets = list(itertools.combinations(repeat_indices, r=N_repeats//2))
        cchalf = 0.
        n_iters = min(len(half_sets)//2, max_iters)
        for i in range(n_iters):
            first_half, second_half = responses[:, half_sets[i], :], responses[:, half_sets[-1-i], :]  # mutually exclusive (itertools)
            psth_first_half, psth_second_half = first_half.mean(dim=1), second_half.mean(dim=1)
            cchalf += correlation_coefficient(psth_first_half, psth_second_half, reduction='None')
        cchalf /= n_iters
        ccmax = torch.sqrt(2 / (1 + 1 / torch.sqrt(torch.pow(cchalf, 2))))
        return ccmax


########################################################
# NORMALIZED METRICS
########################################################

@torch.no_grad()
def sahani_performance(y_pred, y_gt, signal_frac: float = 1.):
    """
    Computes the prediction success of a model's output y_pred (T coeffs, one for each time-bin) with R different
    repetitions of the ground-truth response (T coeffs)

    See Sahani and Linden (2003), "How Linear are Auditory cortical Responses ?", NIPS

    Note 1:
        no assumption of response normalization required.

    Note 2:
        when only one trial is available for the response, the signal cannot be computed by statistical methods, by
        definition.
        to circumvent this limitation, the user can manually set the signal power to an arbitrary fraction of the
        response power.
        By default, we assume the following worst case scenario for performance evaluation and best case scenario
        in terms of signal acquisition. We assume an absolutely clean response, with zero noise. This assumption yields
        a lower bound of the performance. The 'signal_frac' argument corresponds to this case when set to 1., and to a
        pure noise response when set to 0.

    :param y_pred: (B, T) = (B, M) in original notation
    :param y_gt: (B, R, T) = (B, n, M) in original notation
    :return: (B, )
    """
    B, R, T = y_gt.shape
    avg_resp = y_gt.mean(dim=1).unsqueeze(1)                                # (B, 1, T)
    resp_pow = statistical_power(avg_resp)                                  # (B, 1)
    error = ((y_gt - y_pred.unsqueeze(1)) ** 2).mean(dim=1).unsqueeze(1)    # (B, 1, T)
    error_pow = statistical_power(error)                                    # (B, 1)
    if R > 1:
        pow_of_avg_resp = statistical_power(avg_resp)                       # (B, 1)
        avg_pow_per_resp = resp_pow.mean(dim=1).unsqueeze(1)                # (B, 1)
        sig_pow = (1 / (R - 1)) * (R * avg_pow_per_resp - pow_of_avg_resp)  # (B, 1)
    else:
        sig_pow = signal_frac * resp_pow                                    # (B, 1)
    perf = (resp_pow - error_pow) / sig_pow                                 # (B, 1)
    perf = perf.squeeze(1)                                                  # (B,)
    return perf


@torch.no_grad()
def pennington_prediction_correlation(y_pred, y_gt, precomputed_ttrc=None):
    """
    Computes the noise-corrected prediction correlation between a model's output y_pred (T coeffs, one for each
    time-bin), with R different repetitions of the ground-truth response (T coeffs)

    See Pennington et al. (2023), subection "Noise-corrected prediction correlation to evaluate model performance"

    :param y_pred: (B, T)
    :param y_gt: (B, R, T)
    :return: a scalar
    """
    B, R, T = y_gt.shape
    if precomputed_ttrc is not None:
        assert isinstance(precomputed_ttrc, torch.Tensor) and precomputed_ttrc.shape == torch.Size([B])
        ttrc = precomputed_ttrc                                     # (B,)
    else:
        ttrc = compute_TTRC(y_gt)                                   # (B,)

    y_pred = y_pred.unsqueeze(1).repeat(1, R, 1)                    # (B, R, T)
    y_pred = y_pred.flatten(start_dim=0, end_dim=1)                 # (B*R, T) to treat repeats as batches
    y_gt = y_gt.flatten(start_dim=0, end_dim=1)                     # (B*R, T) idem
    cc = correlation_coefficient(y_pred, y_gt, reduction="None")    # (B*R,) raw cc between prediction and each trial response
    cc = cc.unflatten(dim=0, sizes=(B, R))                          # (B, R)
    mean_cc = cc.mean(dim=1)                                        # (B,)  avg cc over trials
    return mean_cc / torch.sqrt(abs(ttrc))


@torch.no_grad()
def normalized_correlation_coefficient(y_pred, y_gt, precomputed_ccmaxes=None, ccmax_iters=126):
    """
    Computes the average correlation coefficient between two batches of 1D vectors.
    It is computed independently for each pair in the batch, and then averaged over batch dimension.

    :param y_pred: (B, T)
    :param y_gt: (B, R, T)
    :param ccmax_iters: int, the number of combinations of trials to compute the ccmax
    :return: a scalar
    """
    B, R, T = y_gt.shape
    if precomputed_ccmaxes is not None:
        assert isinstance(precomputed_ccmaxes, torch.Tensor) and precomputed_ccmaxes.shape == torch.Size([B])
        ccmax = precomputed_ccmaxes                                     # (B,)
    else:
        ccmax = compute_CCmax(y_gt, max_iters=ccmax_iters)              # (B,)

    mean_resp = y_gt.mean(dim=1)        # average response over repeats --> (B, T)
    cc = correlation_coefficient(y_pred, mean_resp, reduction='None')   # (B,)
    cc_norm = cc / ccmax                                                # (B,)
    return cc_norm
