# Installation instructions

First clone and go inside the repository with:
```shell
git clone https://github.com/urancon/deepSTRF.git
```

We recommend dedicating a new virtual environment to this library, using Anaconda for instance:
```shell
conda create --name neuralfit python=3.10
conda activate neuralfit
```

Then install dependencies:
```shell
pip3 install -r deepSTRF/requirements.txt
```

If you want to have access to deepSTRF from anywhere on your machine, install it with the following command line:
```shell
pip3 install -e deepSTRF
```

To check if the install has worked correctly, open a python shell and try printing the version:
```shell
python3 -c "import deepSTRF; print(deepSTRF.__version__)"
```
