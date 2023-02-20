import os
import torch
import numpy as np
from tqdm import tqdm
from utils import CONST_EPS



class OnlineReplayBuffer:
    _device: torch.device
    _state: np.ndarray
    _action: np.ndarray
    _reward: np.ndarray
    _next_state: np.ndarray
    _next_action: np.ndarray
    _not_done: np.ndarray
    _return: np.ndarray
    _size: int


    def __init__(
        self, 
        device: torch.device, 
        state_dim: int, action_dim: int, max_size: int
    ) -> None:

        self._device = device

        self._state = np.zeros((max_size, state_dim))
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._next_state = np.zeros((max_size, state_dim))
        self._next_action = np.zeros((max_size, action_dim))
        self._not_done = np.zeros((max_size, 1))
        self._return = np.zeros((max_size, 1))
        self._advantage = np.zeros((max_size, 1))

        self._size = 0


    def store(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_p: np.ndarray,
        a_p: np.ndarray,
        not_done: bool
    ) -> None:

        self._state[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._next_state[self._size] = s_p
        self._next_action[self._size] = a_p
        self._not_done[self._size] = not_done
        self._size += 1


    def compute_return(
        self, gamma: float
    ) -> None:

        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc='Computing the returns'):
            self._return[i] = self._reward[i] + gamma * pre_return * self._not_done[i]
            pre_return = self._return[i]


    def compute_advantage(
        self, gamma:float, lamda: float, value
    ) -> None:
        delta = np.zeros_like(self._reward)

        pre_value = 0
        pre_advantage = 0

        for i in tqdm(reversed(range(self._size)), 'Computing the advantage'):
            current_state = torch.FloatTensor(self._state[i]).to(self._device)
            current_value = value(current_state).cpu().data.numpy().flatten()

            delta[i] = self._reward[i] + gamma * pre_value * self._not_done[i] - current_value
            self._advantage[i] = delta[i] + gamma * lamda * pre_advantage * self._not_done[i]

            pre_value = current_value
            pre_advantage = self._advantage[i]

        self._advantage = (self._advantage - self._advantage.mean()) / (self._advantage.std() + CONST_EPS)


    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, self._size, size=batch_size)

        return (
            torch.FloatTensor(self._state[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._next_state[ind]).to(self._device),
            torch.FloatTensor(self._next_action[ind]).to(self._device),
            torch.FloatTensor(self._not_done[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._advantage[ind]).to(self._device)
        )



class OfflineReplayBuffer(OnlineReplayBuffer):

    def __init__(
        self, device: torch.device, 
        state_dim: int, action_dim: int, max_size: int
    ) -> None:
        super().__init__(device, state_dim, action_dim, max_size)


    def load_dataset(
        self, dataset: dict
    ) -> None:
        
        self._state = dataset['observations'][:-1, :]
        self._action = dataset['actions'][:-1, :]
        self._reward = dataset['rewards'].reshape(-1, 1)[:-1, :]
        self._next_state = dataset['observations'][1:, :]
        self._next_action = dataset['actions'][1:, :]
        self._not_done = 1. - (dataset['terminals'].reshape(-1, 1)[:-1, :] | dataset['timeouts'].reshape(-1, 1)[:-1, :])

        self._size = len(dataset['actions']) - 1


    def normalize_state(
        self
    ) -> tuple:

        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        return (mean, std)