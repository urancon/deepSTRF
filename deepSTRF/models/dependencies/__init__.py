"""Vendored third-party backbones used by ``deepSTRF.models.audio.StateNet``.

Only the modules with no maintained PyTorch package on PyPI live here:

- ``s4`` — S4 / S4D, the official ``state-spaces/s4`` *standalone* file,
  which upstream distributes as a copy-into-your-repo single module rather
  than a pip package.
- ``lmu`` — the Legendre Memory Unit from ``hrshtv/pytorch-lmu`` (GitHub-only;
  the PyPI ``lmu`` / ``keras-lmu`` packages are TensorFlow/Keras).

Both rely solely on dependencies already required by deepSTRF
(``numpy`` / ``scipy`` / ``einops``); S4's optional CUDA/pykeops kernels fall
back to a pure-PyTorch path. The Mamba backbone is *not* vendored — it uses
the upstream ``mambapy`` package (a default dependency).
"""
