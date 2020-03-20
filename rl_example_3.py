#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/18/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
# Try reacher2d that is batched inside the environment.
# I.e. all parallelism is handled by environment that takes [BxActDim] actions and returns [BxObsDims] observations.
# The environment spaces are however returned for the single batch.

import datetime
import os

import numpy as np
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.base import *
from rlpyt.algos.pg.ppo import *
from rlpyt.agents.pg.categorical import *
from rlpyt.models.mlp import MlpModel
import matplotlib.pyplot as plt
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch.nn.functional as F
from agents_nn import AgentPgDiscrete
from samplers import BatchedEpisodicSampler

plt.switch_backend('Qt5Agg')
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fancybox'] = True


class MyEnv(Env):

    def __init__(self, batch_T, batch_B) -> None:
        super().__init__()
        self.batch_T = batch_T
        self.batch_B = batch_B
        self.state = None
        self.current_goal = None
        self.iter = 0
        self.action_discrete_mapping = np.array([
            [0.0, 0.0],
            [-0.1, 0.0],
            [0.0, -0.1],
            [0.1, 0.0],
            [0.0, 0.1],
        ])
        self._action_space = IntBox(low=0, high=len(self.action_discrete_mapping))
        self._observation_space = FloatBox(low=-1., high=1., shape=4)  # current state and goal
        self.goal_space = FloatBox(low=-1., high=1., shape=(self.batch_B, 2))

    def get_obs(self):
        return np.concatenate([self.state, self.current_goal], axis=-1).astype(np.float32)

    def step(self, action):
        self.iter += 1
        self.state += self.action_discrete_mapping[action]
        dist = np.linalg.norm(self.state - self.current_goal, axis=-1)
        rew = np.exp(-0.5 * dist) / self.horizon
        return EnvStep(self.get_obs(), rew, self.iter == self.horizon, EnvInfo())

    def reset(self):
        self.state = np.zeros((self.batch_B, 2))
        self.current_goal = self.goal_space.sample()
        self.iter = 0
        return self.get_obs()

    @property
    def horizon(self):
        return self.batch_T


def load_params(log_dir, run_id, test_date):
    if test_date is None:
        exps = os.listdir(log_dir)
        exps.sort(reverse=True)
        test_date = exps[0]
        print('Using the latest experiment with timestamp: {}'.format(test_date))
    params_path = '{}{}/run_{}/params.pkl'.format(log_dir, test_date, run_id)
    return torch.load(params_path)


def plot_obs(observations, max_obs=5):
    from matplotlib.cm import get_cmap
    cmap = get_cmap('tab20')
    colors = cmap.colors
    fig, axes = plt.subplots(1, 1, squeeze=False)
    ax = axes[0, 0]
    for i in range(np.minimum(observations.shape[1], max_obs)):
        c = colors[i % 20]
        ax.plot(observations[0, i, 2], observations[0, i, 3], 'o', label='Goal', color=c)
        ax.plot(observations[0, i, 0], observations[0, i, 1], 'x', label='Start', color=c)
        ax.plot(observations[:, i, 0], observations[:, i, 1], '-', label='Path', color=c)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()


def build_and_train(run_id=0, greedy_eval=False, test=True, test_date=None):
    sampler = BatchedEpisodicSampler(
        EnvCls=MyEnv,
        env_kwargs=dict(),
        batch_T=500,
        batch_B=64,
    )
    log_dir = "data/rl_example_3/"
    init_agent = None
    if test:
        data = load_params(log_dir, run_id, test_date)
        init_agent = data['agent_state_dict']
    runner = MinibatchRl(
        algo=PPO(entropy_loss_coeff=0., learning_rate=3e-4),
        agent=AgentPgDiscrete(greedy_eval,
                              model_kwargs={
                                  'policy_hidden_sizes': [64, 64],
                                  'value_hidden_sizes': [64, 64],
                              },
                              initial_model_state_dict=init_agent,
                              ),
        sampler=sampler,
        n_steps=int(400 * sampler.batch_size),
        log_interval_steps=int(10 * sampler.batch_size),
    )
    if test:
        runner.startup()
        sampler.obtain_samples(0, 'eval')
        obs = sampler.samples_np.env.observation
        plot_obs(obs)
    else:
        with logger_context("{}{}".format(log_dir, datetime.datetime.today().strftime("%Y%m%d_%H%M")),
                            run_id, 'Reacher2D', snapshot_mode="last", use_summary_writer=True, override_prefix=True):
            runner.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--test_date', type=str, default=None)
    parser.add_argument('--greedy_eval', dest='greedy_eval', action='store_true')
    args = parser.parse_args()
    build_and_train(run_id=args.run_id, greedy_eval=args.greedy_eval, test=args.test, test_date=args.test_date)
