.. deepSTRF documentation master file, created by
   sphinx-quickstart on Fri Apr  4 15:24:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

deepSTRF documentation
======================

**deepSTRF** is a PyTorch library for **system identification of sensory neurons** —
predicting trial-resolved neural responses (spikes, calcium fluorescence, EEG,
intracellular potential, ...) from naturalistic stimuli with deep neural network
models. It bundles a growing zoo of publicly available recordings, a unified
four-slot model template, pretrained checkpoints on the
`Hugging Face Hub <https://huggingface.co/urancon>`_, a NaN-aware metrics suite,
and a thin opt-in training utility.

This site is the project's reference documentation. Source code, issue tracker,
and pull requests live on `GitHub <https://github.com/urancon/deepSTRF>`_.

.. note::

   The current release focuses on **auditory** datasets and models. The video API
   base classes are exposed under ``deepSTRF.datasets.video`` and
   ``deepSTRF.models.video``, but the specific video loaders are parked on the
   `archive/video-api-v0
   <https://github.com/urancon/deepSTRF/tree/archive/video-api-v0>`_ branch and
   will be revived once rewritten against the modernized base class.


Getting started
---------------

If this is your first time here, the fastest path is:

#. Install with ``pip install -e ".[dev]"`` — see :doc:`_source/md/README_installation`.
#. Skim the four design notes that define deepSTRF's public contracts:
   :doc:`data <_source/md/data_paradigm>`,
   :doc:`models <_source/md/model_paradigm>`,
   :doc:`metrics <_source/md/metrics_paradigm>`,
   and :doc:`fitter <_source/md/fitter>`.
   They are short and compose into the entire API surface.
#. Open one of the runnable example notebooks — for instance
   :doc:`crcns_aa_tutorial <_source/ipynb/crcns_aa_tutorial>` to load a dataset
   end-to-end, or
   :doc:`load_pretrained_statenet_ns1 <_source/ipynb/load_pretrained_statenet_ns1>`
   to grab a published Hugging Face checkpoint and score it. Each notebook opens
   in Colab in one click.


Tour of the documentation
-------------------------

The sidebar groups pages into five sections:

**Quickstart**
   Installation, the four design notes that define the data / model / metrics /
   training contracts, a short walkthrough, supported file formats, and the
   publications index.

**Datasets**
   The dataset zoo, split by modality. Each dataset has its own page with the
   original publication, licensing, the response shape, and instructions on how
   to obtain the data (auto-download where possible).

**Models**
   The model zoo, the standalone AdapTrans module documentation, and the
   registry of pretrained checkpoints we publish on the Hugging Face Hub.

**Examples**
   Runnable notebooks, also available under the ``examples/`` folder of the
   repo and one click from Colab. Grouped into *Start here* (the gentlest
   end-to-end tutorials), *Dataset inspection* (per-dataset visual
   walkthroughs), and *Advanced / analyses* (parameterized STRFs,
   gradient-attribution receptive fields, learnable front-ends).

**API**
   Auto-generated reference for every public class and function under
   ``deepSTRF.*``.


Citing deepSTRF
---------------

If deepSTRF is useful for your work, please cite the relevant paper (BibTeX in
the `README citation section
<https://github.com/urancon/deepSTRF#-citation>`_):

* Rançon, Masquelier & Cottereau, *A general model unifying the adaptive,
  transient and sustained properties of ON and OFF auditory neural responses*,
  PLOS Computational Biology, **20** (8), 2024.
  `DOI: 10.1371/journal.pcbi.1012288 <https://doi.org/10.1371/journal.pcbi.1012288>`_

* Rançon, Masquelier & Cottereau, *Temporal recurrence as a general mechanism to
  explain neural responses in the auditory system*,
  Communications Biology, **8**:1456, 2025.
  `DOI: 10.1038/s42003-025-08858-3 <https://doi.org/10.1038/s42003-025-08858-3>`_

A running list of downstream publications built on deepSTRF lives on the
:doc:`Publications <_source/md/README_publications>` page — pull requests
welcome to add yours.


.. toctree::
   :maxdepth: 1
   :caption: Quickstart
   :hidden:

   _source/md/README_installation.md
   _source/md/data_paradigm.md
   _source/md/dataset_concatenation.md
   _source/md/model_paradigm.md
   _source/md/metrics_paradigm.md
   _source/md/fitter.md
   _source/md/logging.md
   _source/md/README_formats.md
   _source/md/README_publications.md


.. toctree::
   :maxdepth: 2
   :caption: Datasets
   :hidden:

   _source/md/README_datasets.md
   _source/md/README_video_datasets.md
   _source/md/README_audio_datasets.md


.. toctree::
   :maxdepth: 2
   :caption: Models
   :hidden:

   _source/md/README_models.md
   _source/md/README_AdapTrans.md
   _source/md/wav2spec.md
   _source/md/README_gradmap_strf.md
   _source/md/pretrained_models.md


.. toctree::
   :maxdepth: 1
   :caption: Examples — Start here
   :hidden:

   _source/ipynb/crcns_aa_tutorial.ipynb
   _source/ipynb/fit_ns1_statenet.ipynb
   _source/ipynb/load_pretrained_statenet_ns1.ipynb
   _source/ipynb/dataset_concatenation.ipynb
   _source/ipynb/fit_ns1_linear_from_waveform.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples — Dataset inspection
   :hidden:

   _source/ipynb/aa4_inspection.ipynb
   _source/ipynb/inspect_crcns_ac1.ipynb
   _source/ipynb/explore_nat4.ipynb
   _source/ipynb/explore_downer2025.ipynb
   _source/ipynb/explore_wingert2026.ipynb
   _source/ipynb/alice_eeg_tutorial.ipynb
   _source/ipynb/le_2025_baseline.ipynb
   _source/ipynb/espejo_nat_nrf.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples — Advanced / analyses
   :hidden:

   _source/ipynb/strf_parameterizations_ns1.ipynb
   _source/ipynb/strf_gradmap_aa2.ipynb
   _source/ipynb/adaptrans_transformer_aa1.ipynb
   _source/ipynb/learnable_frontend_ns1.ipynb

.. toctree::
   :maxdepth: 3
   :caption: API
   :hidden:

   _source/modules.rst
