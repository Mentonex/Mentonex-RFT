"""Common utils for algorithm module.

Modified from https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py
"""


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)


def prefix_metrics(src_metrics: dict, prefix: str, dst_metrics: dict = None) -> dict:
    if dst_metrics is None:
        dst_metrics = {}
    for k, v in src_metrics.items():
        dst_metrics[f"{prefix}/{k}"] = v
    return dst_metrics
