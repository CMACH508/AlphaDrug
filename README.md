<!--
 * @Author: QHGG
 * @Date: 2022-05-07 17:24:57
 * @LastEditTime: 2022-08-22 18:17:11
 * @LastEditors: QHGG
 * @Description: 
 * @FilePath: /AlphaDrug/README.md
-->
## AlphaDrug — Official PyTorch Implementation

Traditional drug discovery is very laborious, expensive, and time-consuming, due to the huge combinatorial complexity of the discrete molecular search space. Researchers have turned to machine learning methods for help to tackle this difficult problem. However, most existing methods are either virtual screening on the available database of compounds by protein-ligand affinity prediction, or unconditional molecular generation which does not take into account the information of the protein target. 
In this paper, we propose a protein target-oriented de novo drug design method, called AlphaDrug. Our method is able to automatically generate molecular drug candidates in an autoregressive way, and the drug candidates can dock into the given target protein well. To fulfill this goal, we devise a modified transformer network for the joint embedding of protein target and the molecule, and a Monte Carlo Tree Search (MCTS) algorithm for the conditional molecular generation. In the transformer variant, we impose a hierarchy of skip connections from protein encoder to molecule decoder for efficient feature transfer.
The transformer variant computes the probabilities of next atoms based on the protein target and the molecule intermediate. We use the probabilities to guide the look-ahead search by MCTS to enhance or correct the next-atom selection. Moreover, MCTS is also guided by a value function implemented by a docking program, such that the paths with many low docking values are seldom chosen. Experiments on diverse protein targets demonstrate the effectiveness of our methods, indicating that AlphaDrug is a potentially promising solution to target-specific de novo drug design.

This repository contains  the **supplementary material** and  the **official PyTorch implementation** of the paper: **AlphaDrug: Protein Target
Specific De Novo Molecular Generation**


## Resources


#### Supplementary material related to our paper is available via the following links:

- Google Drive: [Supplementary Material](https://drive.google.com/drive/folders/1myoeLdsOYz8mSvYEhSdMfUszUJlaJR3u?usp=sharing)

## Datasets

- [train-val-data.tsv](https://drive.google.com/drive/folders/1myoeLdsOYz8mSvYEhSdMfUszUJlaJR3u?usp=sharing): It contains all sequence pairs for training and validation.

- [data/train-val-split.json](https://github.com/CMACH508/AlphaDrug/blob/main/data/train-val-split.json): It contains the index of the training pairs and test pairs in [train-val-data.tsv](https://drive.google.com/drive/folders/1myoeLdsOYz8mSvYEhSdMfUszUJlaJR3u?usp=sharing).

- [data/testing-proteins.txt](https://github.com/CMACH508/AlphaDrug/blob/main/data/testing-proteins-100.txt): It contains all pdbids of the testing proteins which can be downloaded from [PDBbind website](http://www.pdbbind.org.cn/).


## Requirements

### Here we list several key packages as follows:
| Name | Version | Build | Channel |
| :-----| :---- | :---- | :---- |
| python | 3.7.10 | hffdb5ce_100_cpython | `http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge` |
| torch | 1.4.0 | pypi_0 | pypi |
| pandas | 1.3.4 | pypi_0 | pypi |
| numpy | 1.21.4 | pypi_0 | pypi |
| smina | 2020.12.10 | h37f9cb6_0 | conda-forge |
| rdkit | 2020.09.5 | py37he53b9e1_0 | conda-forge |
| mmseqs2 | 13.45111 | h95f258a_1 | bioconda |
| openbabel | 3.1.1 | py37h200e996_1 | conda-forge |
| biopython | 1.79 | pypi_0 | pypi |

## Model Training

- Before training, please download [train-val-data.tsv](https://drive.google.com/drive/folders/1myoeLdsOYz8mSvYEhSdMfUszUJlaJR3u?usp=sharing) to the data folder.

- There are several key args for training listed as follows:
    | Argument | Description | Default | Type |
    | :-----| :---- | :---- | :---- |
    | --layers | Number of layers in transformer | 4 | int |
    | --bs | Batch size | 32 | int |

- Train lmser transformer:

    ```shell
    cd your_project_path
    python train.py --layers 4 --bs 32 --device 0,1,2,3
    ```

## Pretrained Model

### We provide three pretrained models, i.e., LT, T and TE, as follows:
| Model  | Path |
| :----- | :---- | 
| Lmser Transformer | ./experiment/LT/model/30.pt|
| Original Transformer ([Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)) | ./experiment/T/model/30.pt|
| Transformer Encoder | ./experiment/TE/model/30.pt|

## Run Beam Search (BS)

### There are several key args for BS listed as follows:
| Argument | Description | Default | Type |
| :-----| :---- | :---- | :---- |
| -k | Protein index | 0 | int |
| -bs | Beam size in BS| 10 | int |
| -p | NN model path| LT | str |

### Here is an example of running beam search on protein 1a9u with a beam size of 10 using the pretrained model LT.
```shell
cd your_project_path
python beamsearch.py -k 0 -bs 10 -p LT
```

## Run Monte Carlo Tree Search (MCTS)

### There are several key args for MCTS listed as follows:
| Argument | Description | Default | Type |
| :-----| :---- | :---- | :---- |
| -k | Protein index | 0 | int |
| -st | Number of simulation times in MCTS| 50 | int |
| -p | NN model path | LT | str |
| --max | max mode or freq mode | True | bool |

### Here is an example of running MCTS on protein 1a9u with 50 simulation times using the pretrained model LT in max mode.
```shell
cd your_project_path
python mcts.py -k 0 -st 50 -p LT --max
```