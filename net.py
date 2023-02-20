import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x


def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    final_activation: str
) -> torch.nn.modules.container.Sequential:

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)



class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, state_dim: int, hidden_dim: int, depth: int
    ) -> None:
        super().__init__()
        self._net = MLP(state_dim, hidden_dim, depth, 1, 'relu')


    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)



class QMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, 
        state_dim: int, action_dim: int, hidden_dim: int, depth:int
    ) -> None:
        super().__init__()
        self._net = MLP((state_dim + action_dim), hidden_dim, depth, 1, 'relu')


    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=1)
        return self._net(sa)



class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, 
        state_dim: int, hidden_dim: int, depth: int, action_dim: int, 
    ) -> None:
        super().__init__()
        self._net = MLP(state_dim, hidden_dim, depth, (2 * action_dim), 'tanh')
        self._log_std_bound = (-5., 0.)


    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist