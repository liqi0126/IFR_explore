# IFR-Explore

**IFR-Explore: Learning Inter-object Functional Relationships in 3D Indoor Scenes**

**ICLR 2022**

[Qi Li](https://liqi17thu.github.io/), [Kaichun Mo](https://cs.stanford.edu/~kaichun/), [Yanchao Yang](https://yanchaoyang.github.io/), [Hang Zhao](http://people.csail.mit.edu/hangzhao/), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/)

**Abstact:**
Building embodied intelligent agents that can interact with 3D indoor environments has received increasing research attention in recent years. While most works focus on single-object or agent-object visual functionality and affordances, our work proposes to study a novel, underexplored, kind of visual relations that is also important to perceive and model -- inter-object functional relationships (e.g., a switch on the wall turns on or off the light, a remote control operates the TV). Humans often spend no effort or only a little to infer these relationships, even when entering a new room, by using our strong prior knowledge (e.g., we know that buttons control electrical devices) or using only a few exploratory interactions in cases of uncertainty (e.g., multiple switches and lights in the same room). In this paper, we take the first step in building AI system learning inter-object functional relationships in 3D indoor environments with key technical contributions of modeling prior knowledge by training over large-scale scenes and designing interactive policies for effectively exploring the training scenes and quickly adapting to novel test scenes. We create a new dataset based on the AI2Thor and PartNet datasets and perform extensive experiments that prove the effectiveness of our proposed method.

## Installation
1. Install [Pytorch](https://pytorch.org/get-started/locally/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and [Pointnet2_Pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
2. Run `pip install -r requirements.txt` to install required packages
3. Run `pip install -e gym-IFR` to install gym env for IFR env.

## Preparation

1. Download [dataset](https://drive.google.com/drive/folders/1oLIiiMggOWkgj9O-adIq7YoR4mnHmAHC?usp=sharing).
2. Download [pretrained pointnet](https://drive.google.com/drive/folders/1wgB1PUGZRGq_BigulIQs7MFrHSa92JSO?usp=sharing).
3. Download [test dataset](https://drive.google.com/drive/folders/1zR7_LRJBzP4XJin68MuAOshuZDl8nKU9?usp=sharing).

## Usage

### Large-scale exploration

To explore bedroom scenes, run
```
python exploration.py --output_dir [path for output] --use_scene_attr --bedroom
```

### Fast adaption

Run 
```
python fast_adaptation.py --prior_path [ckp for prior net] --pred_path [ckp for pred net] --use_scene_attr --eval_path [test scenes for eval]
```

Add argument `--vis_path [path for visualization]` if visualization is needed.

## Citation
```
@inproceedings{
li2022ifrexplore,
title={{IFR}-Explore: Learning Inter-object Functional Relationships in 3D Indoor Scenes},
author={QI LI and Kaichun Mo and Yanchao Yang and Hang Zhao and Leonidas Guibas},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=OT3mLgR8Wg8}
}
```
