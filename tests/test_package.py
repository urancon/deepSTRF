"""Smoke tests: the package imports cleanly and exposes its public version."""

import importlib
import re

import pytest


def test_package_imports():
    import deepSTRF  # noqa: F401


def test_version_is_a_pep440_string():
    import deepSTRF

    assert isinstance(deepSTRF.__version__, str)
    # Loose PEP 440 check — bounded integer components with optional pre/dev suffix.
    assert re.match(r"^\d+(\.\d+)*([a-zA-Z]+\d*)?$", deepSTRF.__version__), (
        f"__version__ {deepSTRF.__version__!r} is not PEP 440-ish"
    )


def test_version_matches_pyproject():
    """deepSTRF.__version__ must stay in sync with the pyproject dynamic version."""
    import deepSTRF

    # Resolve project root = parent of tests/
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent

    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError:
        import tomli as tomllib  # py3.10 fallback

    with open(root / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)

    # We use dynamic version, so no static "version" key — just assert it's declared dynamic
    # and that the attr-source matches deepSTRF._version.__version__.
    assert "version" in cfg["project"].get("dynamic", []), (
        "pyproject 'version' should be dynamic"
    )
    dyn = cfg.get("tool", {}).get("setuptools", {}).get("dynamic", {})
    assert dyn.get("version", {}).get("attr") == "deepSTRF._version.__version__"
    assert deepSTRF.__version__ == importlib.import_module(
        "deepSTRF._version"
    ).__version__


CORE_SUBMODULES = [
    "deepSTRF.datasets.neural_dataset",
    "deepSTRF.datasets.audio.audio_dataset",
    "deepSTRF.datasets.video.video_dataset",
    "deepSTRF.models.neural_model",
    "deepSTRF.models.audio.audio_model",
    "deepSTRF.models.video.video_model",
    "deepSTRF.models.layers",
    "deepSTRF.metrics.performance",
]


@pytest.mark.parametrize("modname", CORE_SUBMODULES)
def test_core_submodule_imports(modname):
    importlib.import_module(modname)
