## DeepSleep 2.0: Automated Sleep Arousal Segmentation via Deep Learning

A 300-second example of a 13-channel physiological recording and the corresponding sleep arousal prediction/target labels.

![sample_300s_example_animation](https://user-images.githubusercontent.com/78231009/151840974-9f2d3a59-5499-4823-ac87-f0d26d362ae8.gif)

## Overview
[DeepSleep 2.0](https://www.mdpi.com/2673-2688/3/1/10) is a compact version of [DeepSleep](https://www.nature.com/articles/s42003-020-01542-8), a state-of-the-art, U-Net-inspired, fully convolutional deep neural network, which achieved the highest unofficial score in the [2018 PhysioNet Computing Challenge](https://physionet.org/content/challenge-2018/1.0.0/). The proposed network architecture has a compact encoder/decoder structure containing only 740,551 trainable parameters. The input to the network is a full-length multi-channel polysomnographic recording signal. The network has been designed and optimized to efficiently predict non-apnea sleep arousals on held-out test data at a 5-millisecond resolution level, while not compromising the prediction accuracy. When compared to DeepSleep, the obtained experimental results in terms of gross area under the precision-recall curve (AUPRC) and gross area under the receiver operating characteristic curve (AUROC) suggest that a lightweight architecture, which can achieve similar prediction performance at a lower computational cost, is realizable.

## Requirements
It is assumed that you have the full or partial [PhysioNet](https://physionet.org/content/challenge-2018/1.0.0/) dataset (~135 GB of data per folder) on the disk. In `./data`, you can find two bash scripts to download the PhysioNet dataset.

## Running the code
Here are the essential steps to sucesfully run the main Jupyter notebook file ([`deep_sleep2.ipynb`](deep_sleep2.ipynb)).

**STEP 0: Clone the Repository**

```
 git clone https://github.com/rfonod/deepsleep2.git
 cd deepsleep2
```

**STEP 1: Installation**  

1. Install [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/get-started/locally/). Python 3.8 and PyTorch 1.8.1 were considered for the reported results in the DeepSleep 2.0 paper 
2. [OPTIONAL] Create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) with a specific version of Python
3. Install Python dependencies listed in `requirements.txt`. You can run: 
```
pip3 install -r requirements.txt
```
4. If you plan to use GPU computations (recommended), install [CUDA](https://developer.nvidia.com/cuda-downloads)

**STEP 2: Hyperparameters**
 
A correctly set up `hyperparameters.txt` file must be present in a subdirectory of `./models`. The subdirectory name is specified in the `MODEL_NAME` variable.

**STEP 3: Notebook File**

Run the cells of [`deep_sleep2.ipynb`](deep_sleep2.ipynb) in a sequential order. Consider the description of the **Main Switches** section.

## Citation

If you use this code in your research, please cite the following publication:

```
@Article{Fon22a,
  author    = {Fonod, Robert},
  title     = {{DeepSleep 2.0: Automated Sleep Arousal Segmentation via Deep Learning}},
  journal   = {AI},
  year      = {2022},
  volume    = {3},
  number    = {1},
  pages     = {164-179},
  doi       = {https://doi.org/10.3390/ai3010010},
  publisher = {MDPI},
}
```

Consider also citing the original [**DeepSleep paper**](https://www.nature.com/articles/s42003-020-01542-8).
