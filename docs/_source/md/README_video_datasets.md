## Visual datasets

The first release of deepSTRF focuses on **auditory** datasets and models. A
working video API is on the roadmap but not yet shipped on `develop`; the
`deepSTRF.datasets.video` namespace currently exposes only the
`VideoNeuralDataset` base class so the audio/video module layout mirrors
cleanly.

Earlier drafts of video dataset loaders (Allen Ophys, Allen Ecephys,
CRCNS PVC1/PVC11, CRCNS MT1/MT2, CRCNS VIM2, MICrONS, UW Neural Data
Challenge) have been parked on the `archive/video-api-v0` branch and will be
revived, rewritten against the modernized base class, and merged back as the
video API stabilizes.
