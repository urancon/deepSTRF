# Installation instructions

deepSTRF requires **Python ≥ 3.10**.

## From PyPI (recommended)

Install the latest release and its dependencies in one command:
```shell
pip install deepSTRF
```

We recommend a dedicated virtual environment, e.g. with Anaconda:
```shell
conda create --name deepstrf python=3.10
conda activate deepstrf
pip install deepSTRF
```

### Optional extras

Some functionality has heavier, opt-in dependencies. Install them with the
corresponding extra, e.g. `pip install "deepSTRF[eeg]"`:

- `[docs]` — build the Sphinx documentation
- `[allen]` — Allen Brain Observatory tooling (`allensdk`)
- `[s4]` — JIT-compiled CUDA kernels for the S4 model
- `[eeg]` — MNE for parsing `.fif` EEG files (Alice EEG dataset)
- `[le]` — gammatone filter bank for the Le 2025 dataset

## From source (development)

To work on deepSTRF itself, clone the repository and install it in editable
mode with the development extra (test runner, linters, notebook tooling):
```shell
git clone https://github.com/urancon/deepSTRF.git
cd deepSTRF
pip install -e ".[dev]"      # or `pip install -e .` for runtime only
```

## Check the install

Open a Python shell and print the version:
```shell
python3 -c "import deepSTRF; print(deepSTRF.__version__)"
```
