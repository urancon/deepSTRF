import functools
import inspect
import json
from abc import ABC
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class NeuralModel(nn.Module, ABC):
    """
    Base class for encoding models of sensory neural responses (audio,
    video, ...).

    A four-slot template defines the canonical forward pipeline:

        forward(x):
            x = self.wav2spec(x)        # raw-waveform front-end (future)
            x = self.prefiltering(x)    # AdapTrans / ICAdaptation / Identity
            f = self.core(x)            # shared feature backbone
            return self.readout(f)      # per-neuron projection (B, N, 1, T)

    Concrete subclasses populate the slots in their ``__init__``. Default
    values for ``wav2spec``, ``prefiltering``, ``core`` are ``nn.Identity``,
    so a minimal model only needs to provide a ``readout``. Subclasses
    may override ``forward`` for architectures that don't fit the
    four-slot pipeline (e.g. StateNet's recurrent reshape, Transformer's
    per-frame attention).

    See ``docs/_source/md/model_paradigm.md`` for the full contract.

    Pretrained weights
    ------------------
    Subclasses inherit a HuggingFace Hub interface for pretrained
    checkpoints:

    - :meth:`save_pretrained(dir)`        — write config + weights to a folder
    - :meth:`push_to_hub(repo_id)`        — upload to HF Hub (auth required)
    - :meth:`from_pretrained(repo_or_dir)` — instantiate + load weights

    The init kwargs needed to rebuild the architecture are auto-captured
    on construction (see :meth:`__init_subclass__`), so end-users never
    have to supply a config dict — ``StateNet.from_pretrained("urancon/...")``
    just works.
    """

    def __init__(self, out_neurons: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # number of output neurons N (used by validate() and STRF_gradmap)
        self.O = out_neurons

        # canonical pipeline slots — defaults are no-ops; subclasses override
        self.wav2spec = nn.Identity()
        self.prefiltering = nn.Identity()
        self.core = nn.Identity()
        # readout has no sensible default; subclasses must set it before
        # forward() is called. validate() enforces this.

    def __init_subclass__(cls, **kwargs):
        """Wrap each subclass's ``__init__`` to auto-capture JSON-serialisable
        kwargs into ``self._init_kwargs``.

        The capture only fires for the *leaf* class — ``super().__init__``
        calls cascading through intermediate wrappers no-op. Non-JSON values
        (e.g. an ``nn.Module`` passed for ``prefiltering``) are silently
        dropped; they have to be re-supplied via ``from_pretrained(...,
        extra_kwargs=...)``.
        """
        super().__init_subclass__(**kwargs)
        if "__init__" not in cls.__dict__:
            return
        original_init = cls.__init__
        sig = inspect.signature(original_init)

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kw):
            # Only the leaf class wrapper captures. Intermediate super().__init__
            # cascades land in their own wrappers but type(self) won't match.
            if type(self) is cls:
                self._init_kwargs = _capture_kwargs(sig, self, args, kw)
            original_init(self, *args, **kw)

        cls.__init__ = wrapped_init

    def forward(self, stimulus):
        """Default template forward: wav2spec → prefiltering → core → readout."""
        x = self.wav2spec(stimulus)
        x = self.prefiltering(x)
        f = self.core(x)
        return self.readout(f)

    def detach(self):
        """Detach stateful variables and parameters from the computational graph (cf. spikingjelly)."""
        pass

    def count_trainable_params(self):
        """Return the total number of trainable parameters within the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def validate(self):
        """
        Check that the instance is deepSTRF-compatible.

        Subclasses should call ``super().validate()`` and then add their own
        checks (e.g. :class:`AudioEncodingModel` checks ``F, T > 0``).
        """
        assert isinstance(self.O, int) and self.O > 0, \
            f"self.O must be a positive int (got {self.O!r})"
        assert hasattr(self, 'readout') and isinstance(self.readout, nn.Module), \
            f"{type(self).__name__}.readout must be set to an nn.Module before validate()"
        for slot in ('wav2spec', 'prefiltering', 'core'):
            assert isinstance(getattr(self, slot), nn.Module), \
                f"self.{slot} must be an nn.Module (got {type(getattr(self, slot)).__name__})"

    # ------------------------------------------------------------------
    # Pretrained-weights API (HuggingFace Hub)
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: Union[str, Path], *,
                        metadata: Optional[dict] = None,
                        model_card: Optional[str] = None) -> Path:
        """Write a checkpoint folder (``config.json`` + ``model.safetensors``).

        See :func:`deepSTRF.utils.hub.save_pretrained_to_dir` for the
        full contract.
        """
        from deepSTRF.utils.hub import save_pretrained_to_dir
        return save_pretrained_to_dir(self, save_dir, metadata=metadata,
                                      model_card=model_card)

    @classmethod
    def from_pretrained(cls, repo_id_or_path: Union[str, Path], *,
                        extra_kwargs: Optional[dict] = None,
                        strict: bool = True,
                        map_location: Union[str, torch.device] = "cpu",
                        cache_dir: Optional[Union[str, Path]] = None,
                        revision: Optional[str] = None,
                        token: Optional[str] = None,
                        return_metadata: bool = False):
        """Instantiate ``cls`` from an HF Hub repo or a local checkpoint folder.

        Parameters
        ----------
        repo_id_or_path : str or Path
            ``"<owner>/<name>"`` (HF Hub) or a path to a local folder
            produced by :meth:`save_pretrained`. The decision is made by
            :func:`pathlib.Path.is_dir` — if the path exists locally it
            wins, otherwise we treat the string as a Hub repo id.
        extra_kwargs : dict, optional
            Override / extend the saved config. Use this to re-supply any
            ``__init__`` argument that wasn't JSON-serialisable at save
            time (e.g. a custom ``prefiltering`` module).
        strict : bool, default True
            ``state_dict`` strictness.
        map_location : str or torch.device, default ``'cpu'``
        cache_dir, revision, token
            Forwarded to :func:`huggingface_hub.snapshot_download`.
        return_metadata : bool, default False
            If True, return ``(model, metadata)`` instead of just ``model``.

        Returns
        -------
        model
            Instance of ``cls`` with weights loaded.
        """
        from deepSTRF.utils.hub import download_pretrained, load_pretrained_from_dir
        path = Path(str(repo_id_or_path)).expanduser()
        if path.is_dir():
            local_dir = path
        else:
            local_dir = download_pretrained(str(repo_id_or_path),
                                            cache_dir=cache_dir,
                                            revision=revision, token=token)
        model, metadata = load_pretrained_from_dir(
            cls, local_dir, extra_kwargs=extra_kwargs,
            strict=strict, map_location=map_location,
        )
        if return_metadata:
            return model, metadata
        return model

    def push_to_hub(self, repo_id: str, *,
                    metadata: Optional[dict] = None,
                    model_card: Optional[str] = None,
                    private: bool = False,
                    token: Optional[str] = None,
                    commit_message: Optional[str] = None) -> str:
        """Push this model to ``repo_id`` on the HF Hub.

        Saves the checkpoint to a temporary folder and uploads it via
        :func:`deepSTRF.utils.hub.upload_pretrained`. Creates the repo on
        the fly if it doesn't exist; user must be authenticated with
        write access (``huggingface-cli login`` or ``token=``).

        Returns
        -------
        str
            URL of the resulting commit.
        """
        import tempfile
        from deepSTRF.utils.hub import save_pretrained_to_dir, upload_pretrained
        with tempfile.TemporaryDirectory() as tmp:
            save_pretrained_to_dir(self, tmp, metadata=metadata, model_card=model_card)
            return upload_pretrained(tmp, repo_id, private=private, token=token,
                                     commit_message=commit_message)


def _capture_kwargs(sig: inspect.Signature, self, args: tuple, kw: dict) -> dict:
    """Bind ``args`` / ``kw`` to ``sig`` and keep only JSON-serialisable
    named parameters (skipping ``self`` / ``*args`` / ``**kwargs``)."""
    try:
        bound = sig.bind_partial(self, *args, **kw)
    except TypeError:
        return {}
    bound.apply_defaults()
    captured = {}
    for name, val in bound.arguments.items():
        if name == "self":
            continue
        param = sig.parameters.get(name)
        if param is not None and param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        try:
            json.dumps(val)
        except (TypeError, ValueError):
            continue
        captured[name] = val
    return captured
