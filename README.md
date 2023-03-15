
# Learning Attention Propagation for Compositional Zero-Shot Learning
This is the un-official PyTorch code of the WACV2023 paper [Learning Attention Propagation for Compositional Zero-Shot Learning][https://openaccess.thecvf.com/content/WACV2023/papers/Khan_Learning_Attention_Propagation_for_Compositional_Zero-Shot_Learning_WACV_2023_paper.pdf]. The code provides the implementation of the Cape method.

<p align="center">
  <img src="utils/img.png" />
</p>

## Setup 

1. Clone the repo 

2. We recommend using Anaconda for environment setup. To create the environment and activate it, please run:
```
    conda env create --file environment.yml
    conda activate czsl
```

4. Go to the cloned repo and open a terminal. Download the datasets and embeddings, specifying the desired path (e.g. `DATA_ROOT` in the example):
```
    bash ./utils/download_data.sh DATA_ROOT
    mkdir logs
```

## Training

```
    python train.py --config CONFIG_FILE
```
where `CONFIG_FILE` is the path to the configuration file of the model. 
The folder `configs` contains configuration files for all methods.

To run Cape on MitStates, the command is just:
```
    python train.py --config configs/cape/mit.yml
```
On UT-Zappos, the command is:
```
    python train.py --config configs/cape/utzappos.yml
```
On CGQA, the command is:
```
    python train.py --config configs/cape/cgqa.yml
```

**Note:** To create a new config, all the available arguments are indicated in `flags.py`. 

## Test
 

To test a model, the code is simple:
```
    python test.py --logpath LOG_DIR
```
where `LOG_DIR` is the directory containing the logs of a model.


## References

```
@inproceedings{khan2023learning,
  title={Learning Attention Propagation for Compositional Zero-Shot Learning},
  author={Khan, Muhammad Gul Zain Ali and Naeem, Muhammad Ferjad and Van Gool, Luc and Pagani, Alain and Stricker, Didier and Afzal, Muhammad Zeshan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3828--3837},
  year={2023}
}
```


**Note**: For this code I have taken halp from following github repo:
https://github.com/ExplainableML/czsl
