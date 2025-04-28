from setuptools import setup, find_packages

setup(
    name='deepSTRF',
    version='0.0.1',
    description='A PyTorch-based library and benchmark for fitting sensory neural responses with deep neural network models',
    url="https://github.com/urancon/deepSTRF",
    author='Ulysse Rancon',
    author_email='ulysse.rancon@gmail.com',
    license="GPL-3.0",
    packages=find_packages(include=['deepSTRF',
                                    'deepSTRF.datasets', 'deepSTRF.datasets.audio', 'deepSTRF.datasets.video',
                                    'deepSTRF.models', 'deepSTRF.models.audio', 'deepSTRF.models.video',
                                    'deepSTRF.metrics',
                                    'deepSTRF.utils',
                                    ]),
    install_requires=[
        'numpy==1.23.5',
        'scipy==1.15.2',
        'scikit-image',
        'torch==2.5.1',
        'torchaudio==2.5.1',
        'soundfile',
        'pytorch_lightning==2.5.1',
        'matplotlib==3.10.1',
        'Pillow==11.1.0',
        'torchvision',
        'DCLS==0.1.1',
        'einops==0.8.1',
        'h5py',
        'tables',
        'pandas',
        'wandb',
        'tqdm'
    ],
    extras_require={
        'allen': ['allensdk, xarray'],
        'nems': ['PyNEMS']
    }
)
