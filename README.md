# deepSTRF
*Fitting of auditory neural responses with deep neural network models*

Work in progress - Jan. 2024

____

## Presentation

This repository is associated with the paper "[A general theoretical framework unifying the adaptive, transient and 
sustained properties of ON and OFF auditory neural responses](BIOARXIV_URL)" by Rançon et al.

It contains major codes for result reproduction. In particular, it provides several publicly available datasets that in
convenient **PyTorch** classes, as well as ready-to-deploy computational models and the AdapTrans model of auditory 
ON/OFF responses.

![placeholder.png](docs/homepage_illustration.png)

Details about currently available [models](docs/README_models.md) and [datasets](docs/README_datasets.md) in the doc folder.


## Benchmark

To foster improvement of auditory neural encoding models, we report here information about the best-performing models, 
on each dataset. *Feel free to contact us, if you want to claim a spot on the podium of either dataset !*
A ready-to-deploy PyTorch model class will have to be provided to support your claim, and facilitate the work of future 
researchers.


| **Dataset**  | **Model** | Ranking | **Note**       | **Parameters** | **Perfs <br/>(CCraw / CCnorm) [%]** | **Paper**                                                                                           | 
|--------------|-----------|:-------:|----------------|----------------|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| **NS1**      | 2D-CNN    |   🥇    | uses AdapTrans | 37,276         | 43.7 / 65.3                         | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |          
|              |           |   🥈    |                |                |                                     |                                                                                                     |          
|              |           |   🥉    |                |                |                                     |                                                                                                     |
| **Wehr**     | NRF       |   🥇    | uses AdapTrans | 40,261         | 26.0 / 26.3                         | [Harper et al.](DOI:10.1371/journal.pcbi.1005113)                                                   |          
|              |           |   🥈    |                |                |                                     |                                                                                                     |          
|              |           |   🥉    |                |                |                                     |                                                                                                     |
| **NAT4 A1**  | 2D-CNN    |   🥇    | uses AdapTrans | 15,748         | 32.6 / 56.9                         | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|              |           |   🥈    |                |                |                                     |                                                                                                     |          
|              |           |   🥉    |                |                |                                     |                                                                                                     |
| **NAT4 PEG** | 2D-CNN    |   🥇    | uses AdapTrans | 15,748         | 37.2 / 62.4                         | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|              |           |   🥈    |                |                |                                     |                                                                                                     |
|              |           |   🥉    |                |                |                                     |                                                                                                     |


## Installation

This repository assumes you are comfortable with Python environments and conda. To learn more about conda, please visit
https://anaconda.org/.

First create a conda environment and activate it with the following commands:
```shell
conda create --name deepSTRF_env python=3.8
conda activate deepSTRF_env
```

Then, download this repository and get inside:
```shell
git clone https://github.com/urancon/deepSTRF
cd deepSTRF
```

Install Python libraries and dependencies for this code:
```shell
pip3 install -r requirements.txt
```

Download the datasets and move them to the appropriate place:
```shell
Explain how to download and install the datasets ! 
```

Now you should be all set up to reproduce our experiments. Have fun !


## Train a model

We use [Weights and Biases](https://wandb.ai/) for logging during model training. It is a popular tool among deep learning 
researchers, as it allows to synchronize, plot, and compare metrics for your different runs on a single cloud space, for
free. A downside is that it requires an account; please refer to their website for more information.

The script `main.py` allows you to reproduce major experiments presented in our paper. To train our model with default 
settings, just execute the following command:
```shell
python3 main.py
```

You can also do your own experiment by changing the hyperparameters ! For instance:
```shell
python3 main.py -option1 value1 -option2 value 2
```

To know more about possible options, please do:
```shell
python3 main.py --help
```


## Contributing

In building and maintaining this repository, our goal is to contribute to harmonize the preprocessing of datasets and 
model training procedures. 

We provide some guidelines on the data formats, tensors, models, etc. The automatic differentiation and GPU parallelization
enabled by the PyTorch deep learning library makes a good basis for the task of neural response fitting.

If you agree with the open-science philosophy and would like to share your data, you can either contribute to this 
repository (we would be glad to help you doing so) or make your own with a similar architecture.





## References

This work was made possible by the generous publication of several electrophysiology datasets. If you find them useful
for your research or use them, please do not forget to cite their corresponding article:
* [NS1](datasets/NS1_DRC/README.md)
* [Wehr](datasets/CRCNS_AC1_Wehr/README.md)
* [NAT4](datasets/NAT4/README.md)



## Citation

If you use this repository useful for your research and/or liked our paper, please consider citing it in your work:
```text
@article{rancon2024adaptrans,
    author={Rancon, Ulysse and Masquelier, Timothée and Cottereau, Benoit},
    year={2024},
    month={01},
    pages={19},
    title={A general theoretical framework unifying the adaptive, transient and sustained properties of ON and OFF auditory responses},
    publisher = {Cold Spring Harbor Laboratory},
    journal = {bioRxiv},
    doi={10.1101/2024.01.17.576002},
    URL = {https://www.biorxiv.org/content/early/2024/01/20/2024.01.17.576002},
}
```