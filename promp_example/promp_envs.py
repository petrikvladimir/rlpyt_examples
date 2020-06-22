#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-16
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from rlpyt_utils.actions_utils import *
from rlpyt.envs.base import Env, EnvInfo, EnvStep
from rlpyt.spaces.float_box import FloatBox

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class EnvProMP(Env):
    def __init__(self, horizon=100, weight_are_actions=False) -> None:
        super().__init__()
        self.weight_are_actions = weight_are_actions
        self._horizon = horizon
        self.state = None

        self.start_space = FloatBox(low=0., high=1., shape=2)
        self.start_state = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)

        self.iter = 0

        self._action_space = FloatBox(low=-10, high=10, shape=2)  # new state
        self._observation_space = self.get_obs(shape=True)

    def get_time(self):
        return self.iter / self.horizon

    def get_obs(self, shape=False):
        if shape:
            return FloatBox(low=-10, high=10, shape=3)
        return np.concatenate([[self.get_time()], self.state]).astype(np.float32)

    def step(self, action):
        self.iter += 1
        velocity = np.linalg.norm(action - self.state)
        self.state = action.copy()
        dist = np.linalg.norm(self.state - self.goal)

        rewards = dict()
        rewards['goal'] = 0.9 * np.exp(-0.5 * 10 * dist)
        rewards['vel'] = 0.1 * np.exp(-0.5 * 10 * velocity)
        rewards['col'] = -10 * np.exp(-0.5 * self.colision_dist(self.state)) * self.in_collision(self.state)
        # print(rewards)
        return EnvStep(self.get_obs(), sum(rewards.values()) / self.horizon, self.iter == self.horizon, EnvInfo())

    def colision_dist(self, state):
        return np.linalg.norm(state - [0.5, 0.5])

    def in_collision(self, state):
        return self.colision_dist(state) < 0.2

    def reset(self):
        # while True:
        #     self.start_state = self.start_space.sample()
        #     if not self.in_collision(self.start_state):
        #         break
        self.start_state = np.array([1., 1.], dtype=np.float32)
        self.state = self.start_state.copy()
        self.iter = 0
        return self.get_obs()

    @property
    def horizon(self):
        return self._horizon

    @staticmethod
    def plot_states(observations, actions, max_obs=8):
        cmap = get_cmap('tab20')
        colors = cmap.colors
        fig, axes = plt.subplots(1, 1, squeeze=False)
        ax = axes[0, 0]
        for i in range(np.minimum(observations.shape[1], max_obs)):
            c = colors[i % 20]
            ax.plot(observations[0, i, 1], observations[0, i, 2], 'x', label='Start', color=c)
            ax.plot(observations[:, i, 1], observations[:, i, 2], 'x-', label='Path', color=c)

            ax.plot(actions[0, i, 0], actions[0, i, 1], 'x', label='Start', color=c, alpha=0.2)
            ax.plot(actions[:, i, 0], actions[:, i, 1], '-', label='Path', color=c, alpha=0.2)
        ax.add_artist(plt.Circle((0.5, 0.5), 0.2, color='tab:red', fill=True))
        ax.add_artist(plt.Circle((0., 0.), 0.05, color='tab:green', fill=True))
        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-2, 2)
        ax.axis('equal')
        # ax.legend()
        plt.show()
