"""``deepSTRF.training.auto_config`` — build wandb.config / mlflow.log_params
dicts from a deepSTRF model + Fitter kwargs.

The point: a wandb run is most useful when *every* knob that varies
across runs is captured in ``wandb.config`` so the run table can sort /
filter / colour by it. Writing that dict by hand for each experiment is
boring; this module introspects the common deepSTRF audio-encoding
shape and pulls out the obvious fields.

See ``docs/_source/md/fitter.md`` §4.3.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch.nn as nn


_JSON_FRIENDLY = (int, float, str, bool, type(None))


def auto_config(
    model: nn.Module,
    fitter_kwargs: Optional[Mapping[str, Any]] = None,
    dataset_name: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a JSON-friendly config dict from a deepSTRF model + fitter_kwargs.

    Pulled fields, when present:
        - ``model``               : class name of ``model``
        - ``dataset``             : ``dataset_name`` if given
        - ``n_frequency_bands``   : ``model.F``
        - ``temporal_window_size``: ``model.T``
        - ``out_neurons``         : ``model.O``
        - ``prefiltering``        : class name of ``model.prefiltering``
        - ``core``                : class name of ``model.core``
        - ``readout``             : class name of ``model.readout``
        - ``readout.kernel``      : class name of ``model.readout.kernel`` (if present)
        - ``readout.activation``  : class name of ``model.readout.activation`` (if present)
        - ``output_activation``   : class name of ``model.output_activation`` (if present)
        - Every JSON-friendly ``fitter_kwargs`` entry; non-friendly values
          are stored as their ``repr()`` truncated to 60 chars.
        - Every key/value in ``extra``, overwriting any of the above.

    The result is safe to pass as ``wandb.config`` (no torch tensors / Modules).
    """
    cfg: dict = {"model": type(model).__name__}
    if dataset_name is not None:
        cfg["dataset"] = dataset_name

    for attr, key in (
        ("F", "n_frequency_bands"),
        ("T", "temporal_window_size"),
        ("O", "out_neurons"),
    ):
        if hasattr(model, attr):
            v = getattr(model, attr)
            if isinstance(v, _JSON_FRIENDLY):
                cfg[key] = v

    for attr in ("prefiltering", "core", "readout", "output_activation"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module):
            cfg[attr] = type(sub).__name__

    readout = getattr(model, "readout", None)
    if isinstance(readout, nn.Module):
        for sub_attr in ("kernel", "activation"):
            sub = getattr(readout, sub_attr, None)
            if isinstance(sub, nn.Module):
                cfg[f"readout.{sub_attr}"] = type(sub).__name__

    if fitter_kwargs:
        for k, v in fitter_kwargs.items():
            cfg[k] = v if isinstance(v, _JSON_FRIENDLY) else repr(v)[:60]

    if extra:
        cfg.update(extra)

    return cfg


_DEFAULT_SLUG_FIELDS = (
    ("model", str.lower),
    ("dataset", str.lower),
    ("temporal_window_size", lambda v: f"T{v}"),
    ("n_frequency_bands", lambda v: f"F{v}"),
)


def slug_from_config(
    config: Mapping[str, Any],
    fields: Optional[Sequence] = None,
) -> str:
    """Build a short dash-separated slug like ``'linear-ns1-T9-F34'``.

    Parameters
    ----------
    config
        A config dict (typically from :func:`auto_config`).
    fields
        Sequence of ``(key, formatter)`` pairs. ``formatter`` is a
        callable ``value -> str``. Missing keys are silently skipped.
        Default fields: ``model`` (lowercased), ``dataset`` (lowercased),
        ``temporal_window_size`` (as ``T{v}``), ``n_frequency_bands`` (as
        ``F{v}``).
    """
    fields = fields if fields is not None else _DEFAULT_SLUG_FIELDS
    parts = []
    for key, fmt in fields:
        if key in config:
            parts.append(fmt(config[key]))
    return "-".join(parts)
