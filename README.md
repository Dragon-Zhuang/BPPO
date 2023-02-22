# Behavior Proximal Policy Optimization

Author's Pytorch implementation of <b>[ICLR 2023 paper](https://openreview.net/forum?id=3c13LptpIph&referrer=%5Bthe%20profile%20of%20Kun%20LEI%5D(%2Fprofile%3Fid%3D~Kun_LEI1))</b> **B**ehavior **P**roximal **P**olicy **O**ptimization (BPPO). BPPO uses the loss function from Proximal Policy Optimization (PPO) to improve the behavior policy estimated by behavior cloning. 

## The difference between BPPO and PPO

Compared to the loss function of PPO, BPPO does not introduce any extra constraint or regularization. The only difference is the advantage approximation, corresponding to the code difference between `ppo.py` line 88-89 and `bppo.py` line 151-155. 


## Overview of the Code
The code consists of 7 Python scripts and the file `main.py` contains various parameter settings which are interpreted and described in our paper.
### Requirements
- `torch                         1.12.0`
- `mujoco                        2.2.1`
- `mujoco-py                     2.1.2.14`
- `d4rl                          1.1`
### Running the code
- `python main.py`: trains the network, storing checkpoints along the way.
- `Example`: 
```bash
python main.py --env hopper-medium-v2
```
## Citation 
If you use BPPO, please cite our paper as follows:
```
@inproceedings{
zhuang2023behavior,
title={Behavior Proximal Policy Optimization},
author={Zifeng Zhuang and Kun LEI and Jinxin Liu and Donglin Wang and Yilang Guo},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=3c13LptpIph}
}
```
