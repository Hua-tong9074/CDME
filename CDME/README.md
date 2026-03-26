# CDME: Class-aware Dynamic Multicentric Enhancement for Source-Free Domain Adaptation

## Overview
This repository implements a Source-Free Domain Adaptation (SFDA) framework with multicentric pseudo-labeling and dynamic prototype learning.

## Method
Our framework includes:
- Multicentric pseudo-label initialization
- Dynamic temperature scaling
- Soft pseudo-label learning
- Prototype contrastive learning
- EMA-based prototype updates

## Project Structure
```
.
├── data/
├── scripts/
├── main_source.py
├── main_target.py
├── network.py
├── loss.py
├── utils.py
├── data_list.py
└── README.md
```

## Requirements
- Python 3.x
- PyTorch >= 1.7
- torchvision
- numpy, scipy, sklearn, PIL, tqdm

## Dataset
Download VisDA-C:
https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification

Place under:
```
./data/
```

## Training

### Train source model
```
sh ./scripts/train_source.sh
```

### Adapt to target domain
Edit checkpoint path in:
```
./scripts/train_target.sh
```

Run:
```
sh ./scripts/train_target.sh
```

## Acknowledgement
Based on SHOT:
https://github.com/tim-learn/SHOT

## Citation
```
@article{yourname2025cdme,
  title={Class-aware Dynamic Multicentric Enhancement for Source-Free Domain Adaptation},
  author={Your Name},
  year={2025}
}
```
