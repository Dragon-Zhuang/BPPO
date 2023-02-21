## BPPO, For details, please refer to our <b>[ICLR 2023 paper](https://openreview.net/forum?id=3c13LptpIph&referrer=%5Bthe%20profile%20of%20Kun%20LEI%5D(%2Fprofile%3Fid%3D~Kun_LEI1))</b> 
Author's Pytorch implementation of ICLR2023 paper \textbf{B}ehavior \textbf{P}roximal \textbf{P}olicy \textbf{O}ptimization (BPPO).


## Overview of the Code
The code consists of seven Python scripts and the file `main.py` that contains various parameter settings.
### Needed
torch                         1.12.0

torchaudio                    0.12.0

torchvision                   0.13.0

mujoco                        2.2.1

mujoco-py                     2.1.2.14

d4rl                          1.1
### Running the code
- `python main.py`: trains the network, storing checkpoints along the way.
- `Example`: 
```
python main.py --env hopper-medium-v2 
```
