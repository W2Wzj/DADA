# PIDA: Private Information-guided Source-Free Universal Domain Adaptation

## Introduction
Source-free Universal Domain Adaptation (UniDA) aims to transfer knowledge from a source domain to a target domain without access to the source data or prior knowledge of cross-domain category shifts.

In this paper, we propose Private Information-guided Domain Adaptation (PIDA), a novel source-free framework that explicitly identifies both source-private and target-private information to guide model adaptation. PIDA follows a two-stage process: first, it identifies the private information based on prior class probability distributions to exclude potential private classes; second, to further utilize the identified private information, it introduces two specialized loss functions that enhance the distinction between shared- and private-class samples within the target domain.

## Framework
![alt text](image.png)
## Prerequisites
- python3, pytorch, numpy, scipy, sklearn, tqdm, etc.
- We have presented the our conda environment file in `./environment.yml`.

## Dataset
We conduct extensive experiments on three standard domain adaptation benchmarks:
- Office
- OfficeHome
- VisDA

Please manually download these datasets from the official websites and unzip them to the `./data` folder. We have included as supplementary the Office dataset (only Amazon and Dslr available due to submission size constraints).

The data structure should look like:

```
./data
├── Office
│   ├── Amazon
│   │   └── ...
│   └── Webcam
│       └── ...
├── OfficeHome
│   └── ...
└── VisDA
   └── ...
```
## Step
1. Please prepare the environment first.
2. Please download the datasets from the corresponding official websites, and then unzip them to the `./data` folder.
3. Preparing the source model.
4. Performing the target model adaptation.
## Training
```
# Source Model Preparing
bash ./scripts/train_source_OPDA.sh
#Target Model Adaptation
bash ./scripts/train_target_PIDA_OPDA.sh
```
The code to run AaD, GLC, and LEAD baselines is also available. 

## Project stucture: 
```
├── data/                  # Dataset folder
├── figures/               # Framework and result visualizations
├── scripts/               # Training scripts
│   ├── train_source_OPDA.sh
│   ├── train_source_OSDA.sh
│   ├── train_source_PDA.sh
│   ├── train_target_PIDA_OPDA.sh
│   ├── train_target_PIDA_OSDA.sh
│   └── train_target_PIDA_PDA.sh
├── environment.yml       # Conda environment file
└── README.md
```

## Acknowledgements 
The majority of this code has been adapted from the following papers:
```
@inproceedings{sanqing2024LEAD,
  title={LEAD: Learning Decomposition for Source-free Universal Domain Adaptation},
  author={Qu, Sanqing and Zou, Tianpei and He, Lianghua and RÃ¶hrbein, Florian and Knoll, Alois and Chen, Guang and Jiang, Changjun},
  booktitle={CVPR},
  year={2024},
}
@article{nejjar2024recall,
  title={Recall and Refine: A Simple but Effective Source-free Open-set Domain Adaptation Framework},
  author={Nejjar, Ismail and Dong, Hao and Fink, Olga},
  journal={arXiv preprint},
  year={2024}
}
```


## Contact
- Zhijingwang@std.uestc.edu.cn
