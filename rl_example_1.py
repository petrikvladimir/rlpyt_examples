#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/17/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
# Trivial example to test rlpyt library.

import datetime
import os

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.spaces.float_box import FloatBox
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.base import *
from rlpyt.algos.pg.ppo import *
from rlpyt.agents.pg.gaussian import *
from rlpyt.models.pg.mujoco_ff_model import MujocoFfModel
import matplotlib.pyplot as plt

plt.switch_backend('Qt5Agg')
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fancybox'] = True

model_params = dict(observation_shape=(4,), action_size=2, hidden_sizes=[400, 300],
                    # hidden_nonlinearity=torch.nn.ReLU,
                    hidden_nonlinearity=torch.nn.Tanh,
                    )
horizon = 200


class MyEnv(Env):

    def __init__(self) -> None:
        super().__init__()
        self.state = None
        self.current_goal = None
        self.iter = 0
        self._action_space = FloatBox(low=-0.01, high=0.01, shape=2)
        self._observation_space = FloatBox(low=-1., high=1., shape=4)  # current state and goal
        self.goal_space = FloatBox(low=-1., high=1., shape=2)

    def get_obs(self):
        return np.concatenate([self.state, self.current_goal]).astype(np.float32)

    def step(self, action):
        self.iter += 1
        self.state += action * self._action_space.high[0]
        dist = np.linalg.norm(self.state - self.current_goal)
        rew = np.exp(-0.5 * dist) / self.horizon
        return EnvStep(self.get_obs(), rew, self.iter == self.horizon, EnvInfo())

    def reset(self):
        self.state = np.zeros(2)
        self.current_goal = self.goal_space.sample()
        self.iter = 0
        return self.get_obs()

    @property
    def horizon(self):
        return horizon


def build_and_train(run_id=0):
    sampler = SerialSampler(
        EnvCls=MyEnv,
        env_kwargs=dict(),
        eval_env_kwargs=dict(),
        batch_T=horizon,
        batch_B=64,
        max_decorrelation_steps=0,
        eval_n_envs=64,
        eval_max_steps=int(1e6),
        eval_max_trajectories=64,
    )
    algo = PPO(entropy_loss_coeff=0., learning_rate=3e-4)
    agent = GaussianPgAgent(
        ModelCls=MujocoFfModel,
        model_kwargs=model_params,
    )
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(400 * horizon * 64),
        log_interval_steps=int(10 * horizon * 64),
    )
    log_params = dict()

    log_dir = "data/rl_example_1/{}".format(datetime.datetime.today().strftime("%Y%m%d_%H%M"))
    with logger_context(log_dir, run_id, 'Reacher2D', log_params=log_params, snapshot_mode="last",
                        use_summary_writer=True, override_prefix=True):
        runner.train()


def build_and_test(run_id=0, test_date=None, use_mode=False):
    log_dir = "data/rl_example_1/"
    if test_date is None:
        exps = os.listdir(log_dir)
        exps.sort(reverse=True)
        test_date = exps[0]
        print('Using the latest experiment with timestamp: {}'.format(test_date))

    params_path = '{}{}/run_{}/params.pkl'.format(log_dir, test_date, run_id)
    data = torch.load(params_path)
    agent = GaussianPgAgent(
        ModelCls=MujocoFfModel,
        model_kwargs=model_params,
        initial_model_state_dict=data['agent_state_dict'],
    )
    env = MyEnv()
    agent.initialize(env.spaces)
    agent.eval_mode(0)
    while True:
        observations = []
        obs = env.reset()
        action = env.action_space.sample()
        rew = 0.
        for _ in range(horizon):
            observations.append(obs)
            action, action_info = agent.step(torch.from_numpy(obs).float(),
                                             torch.from_numpy(action).float(),
                                             torch.tensor(rew).float())
            action = action.numpy()
            if use_mode:
                action = action_info.dist_info.mean.numpy()  # use mean of the distribution
            obs, rew, _, _ = env.step(action=action)
        observations = np.stack(observations, axis=0)

        fig, axes = plt.subplots(1, 1, squeeze=False)
        ax = axes[0, 0]
        ax.plot(observations[0, 2], observations[0, 3], 'o', label='Goal', color='tab:green')
        ax.plot(observations[0, 0], observations[0, 1], 'x', label='Start', color='tab:blue')
        ax.plot(observations[:, 0], observations[:, 1], '--k', label='Path')
        plt.legend()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--test_date', type=str, default=None)
    parser.add_argument('--use_mode', dest='use_mode', action='store_true')
    args = parser.parse_args()
    if args.test:
        build_and_test(run_id=args.run_id, test_date=args.test_date, use_mode=args.use_mode)
    else:
        build_and_train(run_id=args.run_id, )
