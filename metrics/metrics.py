import torch
import numpy as np


@torch.no_grad()
def correlation_coefficient(y_pred, y_gt):
    """
    Computes the average correlation coefficient between two batches of 1D vectors.
    It is computed independently for each pair in the batch, and then averaged over batch dimension.

    :param y_pred: (B, T)
    :param y_gt: (B, T)
    :return: a scalar
    """
    cov = covariance(y_pred, y_gt)
    var_pred = torch.var(y_pred, dim=1)
    var_gt = torch.var(y_gt, dim=1)
    cc = cov / torch.sqrt(var_pred * var_gt)
    cc_batch_avg = torch.mean(cc)
    return cc_batch_avg


@torch.no_grad()
def covariance(y_pred, y_gt):
    """
    Computes the average covariance between two batches of 1D vectors.
    It is computed independently for each pair in the batch.

    :param y_pred: (B, T)
    :param y_gt: (B, T)
    :return: (B, )
    """
    B, N = y_pred.shape
    y_pred_avg = torch.mean(y_pred, dim=1)
    y_gt_avg = torch.mean(y_gt, dim=1)
    y_pred_rect = y_pred - y_pred_avg.unsqueeze(1).repeat(1, N)
    y_gt_rect = y_gt - y_gt_avg.unsqueeze(1).repeat(1, N)
    cov = torch.sum(y_pred_rect * y_gt_rect, dim=1) / (N-1)
    return cov

