"""HuggingFace Hub integration for pretrained deepSTRF models.

A deepSTRF "checkpoint" is a folder with three files:

- ``config.json``       — JSON-serialisable ``__init__`` kwargs (auto-captured
                          by :class:`~deepSTRF.models.neural_model.NeuralModel`)
                          plus a ``_model_class`` sentinel for safety checks.
- ``model.safetensors`` — the model's ``state_dict``.
- ``README.md``         — optional model card (free-form markdown shown by
                          HF Hub).
- ``metadata.json``     — optional user metadata (test metrics, dataset
                          name, training config, …). Preserved through
                          round-trips, never used for instantiation.

This module exposes four helpers covering the local and Hub sides of that
layout. The high-level methods on
:class:`~deepSTRF.models.neural_model.NeuralModel`
(``save_pretrained`` / ``from_pretrained`` / ``push_to_hub``) are thin
wrappers over these.

Public surface
--------------
- :func:`save_pretrained_to_dir`     — write a model + config to a folder
- :func:`load_pretrained_from_dir`   — instantiate a class + load weights
- :func:`download_pretrained`        — fetch an HF Hub repo to local cache
- :func:`upload_pretrained`          — push a folder to an HF Hub repo
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

# huggingface_hub + safetensors are required runtime deps (cf. pyproject.toml).
from huggingface_hub import snapshot_download, create_repo, upload_folder
from safetensors.torch import save_file as safetensors_save
from safetensors.torch import load_file as safetensors_load


CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
METADATA_FILE = "metadata.json"
MODEL_CARD_FILE = "README.md"


# ---------------------------------------------------------------------------
# Local save / load
# ---------------------------------------------------------------------------

def save_pretrained_to_dir(
    model: nn.Module,
    save_dir: Union[str, Path],
    *,
    metadata: Optional[dict] = None,
    model_card: Optional[str] = None,
) -> Path:
    """Write a deepSTRF checkpoint to ``save_dir``.

    The model must carry a ``_init_kwargs`` attribute (auto-populated by
    :class:`~deepSTRF.models.neural_model.NeuralModel.__init_subclass__`).
    Everything else is optional.

    Parameters
    ----------
    model : nn.Module
        Model instance with ``_init_kwargs`` set.
    save_dir : path-like
        Destination folder. Created if missing; existing files are
        overwritten.
    metadata : dict, optional
        Free-form, JSON-serialisable user metadata (test metrics, dataset
        name, …). Written to ``metadata.json``.
    model_card : str, optional
        Markdown content for ``README.md``. If ``None``, a minimal default
        card is generated from the class name and config.

    Returns
    -------
    Path
        The resolved ``save_dir``.

    Raises
    ------
    AttributeError
        If ``model._init_kwargs`` is missing — i.e. the class isn't a
        :class:`NeuralModel` subclass, or its ``__init__`` was not run
        through the auto-capture wrapper.
    """
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = getattr(model, "_init_kwargs", None)
    if init_kwargs is None:
        raise AttributeError(
            f"{type(model).__name__} has no _init_kwargs attribute. "
            "save_pretrained requires a NeuralModel subclass whose __init__ "
            "was wrapped by NeuralModel.__init_subclass__ (this is automatic "
            "for any class deriving from NeuralModel)."
        )

    # config.json: the kwargs we'll pass to __init__ on reload, plus a
    # _model_class sentinel. The class is recorded only so from_pretrained
    # can warn if someone tries to load (e.g.) StateNet weights into Linear;
    # it is *not* used to look up a class by name (no dynamic import) — the
    # caller picks the class explicitly.
    config = dict(init_kwargs)
    config["_model_class"] = type(model).__name__
    (save_dir / CONFIG_FILE).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    # model.safetensors: state_dict only. We CPU-detach so the file is
    # device-independent; load_pretrained_from_dir maps to the requested
    # device on load.
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    safetensors_save(state_dict, str(save_dir / WEIGHTS_FILE))

    if metadata is not None:
        (save_dir / METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    card = model_card if model_card is not None else _default_model_card(model, config, metadata)
    (save_dir / MODEL_CARD_FILE).write_text(card)

    return save_dir


def load_pretrained_from_dir(
    model_class: Type[nn.Module],
    load_dir: Union[str, Path],
    *,
    extra_kwargs: Optional[dict] = None,
    strict: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, Optional[dict]]:
    """Instantiate ``model_class`` from a deepSTRF checkpoint folder.

    Parameters
    ----------
    model_class : type
        The concrete class to instantiate (e.g. ``StateNet``).
    load_dir : path-like
        Folder produced by :func:`save_pretrained_to_dir`.
    extra_kwargs : dict, optional
        Override / extend the JSON-loaded config. Useful for kwargs that
        couldn't be JSON-serialised at save time (e.g. ``prefiltering`` or
        ``output_activation`` ``nn.Module`` instances). Merged on top of the
        loaded config; ``None`` is allowed.
    strict : bool, default True
        Passed to ``load_state_dict``. Set to ``False`` to tolerate missing
        / unexpected keys (e.g. when loading a population checkpoint into
        a single-cell model).
    map_location : str or torch.device, default ``'cpu'``
        Device to move tensors to. Mirrors ``torch.load``'s argument.

    Returns
    -------
    (model, metadata) : tuple
        ``model`` is an instance of ``model_class`` with weights loaded.
        ``metadata`` is the parsed ``metadata.json`` if present, else
        ``None``.
    """
    load_dir = Path(load_dir).expanduser().resolve()
    config_path = load_dir / CONFIG_FILE
    weights_path = load_dir / WEIGHTS_FILE
    if not config_path.exists():
        raise FileNotFoundError(f"missing {CONFIG_FILE} in {load_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"missing {WEIGHTS_FILE} in {load_dir}")

    config = json.loads(config_path.read_text())
    saved_class = config.pop("_model_class", None)
    if saved_class is not None and saved_class != model_class.__name__:
        # Mismatch is a user error in 99% of cases, but we don't hard-fail —
        # subclass / rename scenarios are legitimate. Loud warning instead.
        import warnings
        warnings.warn(
            f"checkpoint at {load_dir} was saved as {saved_class!r} but "
            f"{model_class.__name__!r}.from_pretrained was called. "
            f"Continuing — set strict=False if state_dict keys diverge.",
            stacklevel=2,
        )

    if extra_kwargs:
        config = {**config, **extra_kwargs}

    model = model_class(**config)
    state_dict = safetensors_load(str(weights_path), device=str(map_location))
    model.load_state_dict(state_dict, strict=strict)

    metadata = None
    metadata_path = load_dir / METADATA_FILE
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return model, metadata


# ---------------------------------------------------------------------------
# HF Hub download / upload
# ---------------------------------------------------------------------------

def download_pretrained(
    repo_id: str,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Fetch all files of an HF Hub model repo into a local cache folder.

    Thin wrapper around :func:`huggingface_hub.snapshot_download`. The Hub
    client handles caching, resumability, and ETag validation, so repeated
    calls with the same ``repo_id`` are essentially free.

    Parameters
    ----------
    repo_id : str
        E.g. ``"urancon/deepSTRF-statenet-gru-ns1"``.
    cache_dir : path-like, optional
        Override the HF Hub cache root. Defaults to
        ``~/.cache/huggingface/hub``.
    revision : str, optional
        Branch, tag, or commit SHA. Defaults to the repo's default branch.
    token : str, optional
        HF auth token for private repos. Public repos work anonymously.
        For convenience, also picks up ``$HF_TOKEN`` automatically via the
        Hub client.

    Returns
    -------
    Path
        Folder containing the snapshot.
    """
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        revision=revision,
        token=token,
    )
    return Path(local_dir)


def upload_pretrained(
    local_dir: Union[str, Path],
    repo_id: str,
    *,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> str:
    """Push a checkpoint folder to an HF Hub model repo.

    Creates the repo on the fly if it doesn't exist (idempotent — safe to
    call repeatedly). The user must be authenticated with write access to
    ``repo_id`` (run ``huggingface-cli login`` once or pass ``token=``).

    Parameters
    ----------
    local_dir : path-like
        Folder produced by :func:`save_pretrained_to_dir`.
    repo_id : str
        ``"<owner>/<name>"`` — the owner must be your username or an org
        you can write to.
    private : bool, default False
        If creating a new repo, mark it private. Ignored if the repo
        already exists.
    token : str, optional
        HF auth token. Defaults to the cached login token / ``$HF_TOKEN``.
    commit_message : str, optional
        Git commit message on the Hub repo. Defaults to a short message
        with the timestamp.

    Returns
    -------
    str
        URL of the resulting commit.
    """
    local_dir = Path(local_dir).expanduser().resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"{local_dir} is not a directory")

    create_repo(repo_id=repo_id, repo_type="model", private=private,
                exist_ok=True, token=token)

    if commit_message is None:
        from datetime import datetime, timezone
        commit_message = f"Upload deepSTRF checkpoint ({datetime.now(timezone.utc).isoformat(timespec='seconds')})"

    commit_info = upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        repo_type="model",
        token=token,
        commit_message=commit_message,
    )
    # huggingface_hub returns a CommitInfo; .commit_url is the public URL.
    return getattr(commit_info, "commit_url", str(commit_info))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _default_model_card(model: nn.Module, config: dict, metadata: Optional[dict]) -> str:
    """Generate a minimal model card so the Hub repo isn't blank."""
    cls_name = type(model).__name__
    config_view = {k: v for k, v in config.items() if k != "_model_class"}
    lines = [
        "---",
        "library_name: deepSTRF",
        "tags:",
        "- neuroscience",
        "- sensory-encoding",
        "- pytorch",
        "---",
        "",
        f"# {cls_name} (deepSTRF)",
        "",
        "Pretrained checkpoint produced with [deepSTRF](https://github.com/urancon/deepSTRF).",
        "",
        "## Usage",
        "",
        "```python",
        f"from deepSTRF.models.audio import {cls_name}  # adjust import path if needed",
        "",
        f'model = {cls_name}.from_pretrained("<owner>/<repo>")',
        "model.eval()",
        "```",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config_view, indent=2, sort_keys=True),
        "```",
    ]
    if metadata:
        lines += [
            "",
            "## Metadata",
            "",
            "```json",
            json.dumps(metadata, indent=2, sort_keys=True),
            "```",
        ]
    return "\n".join(lines) + "\n"
