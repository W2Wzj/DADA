# Double-Boundary Awareness of Shared-Classes for Source-Free Universal Domain Adaptation

## Introduction
Source-Free Universal Domain Adaptation (SF-UniDA) aims to adapt a pre-trained source model to an unlabeled target domain without access to source data or prior knowledge of cross-domain category shifts.
In this paper, we propose Double-boundary Awareness Domain Adaptation (DADA), a category-driven framework that partitions the target domain label space into shared, potential source-private, and target-private categories.
By labeling target-private samples as unknown and filtering out misassigned source-private samples, DADA enhances the quality of target samples and the reliability of pseudo-labels.
To achieve this, we propose Double-bounded Shared-classes Refinement (DSR) module, which refines shared classes by identifying both source- and target-private categories based on prior class probabilities and the entropy distribution.
Additionally, we incorporate Class-Aware Discriminative Learning (CADL) to improve discrimination between shared and target-private samples across domains.
Extensive experiments conducted on three benchmarks demonstrate that DADA outperforms existing SF-UniDA methods, achieving competitive performance.

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
bash ./scripts/train_target_DADA_OPDA.sh
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
│   ├── train_target_DADA_OPDA.sh
│   ├── train_target_DADA_OSDA.sh
│   └── train_target_DADA_PDA.sh
├── environment.yml       # Conda environment file
└── README.md
```

## Citation
If this codebase is useful for your work, please cite the following papers:
'''
@article{wang2025double,
  title={Double-boundary awareness of shared categories for source-free universal domain adaptation},
  author={Wang, Zhijing and Guo, Ji and Sun, Xu and Luo, Yi and Chen, Aiguo},
  journal={Neurocomputing},
  pages={131473},
  year={2025},
  publisher={Elsevier}
}
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
‘’‘


## Contact
- Zhijingwang@std.uestc.edu.cn
