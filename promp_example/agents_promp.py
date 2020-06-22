#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-16
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import torch
from rlpyt.agents.pg.gaussian import GaussianPgAgent
from rlpyt.models.mlp import MlpModel
import numpy as np
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt_utils.promp.promp import ProMP


class ModelProMP(torch.nn.Module):
    def __init__(self, observation_shape, action_size,
                 policy_hidden_sizes=None, policy_hidden_nonlinearity=torch.nn.Tanh,
                 value_hidden_sizes=None, value_hidden_nonlinearity=torch.nn.Tanh,
                 init_log_std=0., min_std=0.,
                 ):
        super().__init__()
        self.min_std = min_std
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))

        self.promp = ProMP(n_dof=action_size, num_basis_functions=50, position_only=True, cov_w_is_diagonal=True,
                           init_scale_cov_w=0.5, cov_eps=1e-7)
        ref_time = torch.linspace(0., 1.)
        ref_y = torch.stack([1 - ref_time, 1 - ref_time], dim=-1).unsqueeze(-1)  # [T x D x 1]
        self.promp.condition(0., torch.tensor([1., 1.]), 1e-2 ** 2 * torch.eye(2))
        # self.promp.condition(1., torch.tensor([0., 0.]), 1e-3 * torch.eye(2))
        self.promp.set_params_from_reference_trajectories(ref_y, t=ref_time, fixed_cov=0.05 ** 2)

        # policy_hidden_sizes = [400, 300] if policy_hidden_sizes is None else policy_hidden_sizes
        value_hidden_sizes = [400, 300] if value_hidden_sizes is None else value_hidden_sizes
        # self.mu = MlpModel(input_size=input_size, hidden_sizes=policy_hidden_sizes, output_size=action_size,
        #                    nonlinearity=policy_hidden_nonlinearity)
        self.v = MlpModel(input_size=input_size, hidden_sizes=value_hidden_sizes, output_size=1,
                          nonlinearity=value_hidden_nonlinearity, )
        # self._log_std = torch.nn.Parameter((np.log(np.exp(init_log_std) - self.min_std)) * torch.ones(action_size))

    @property
    def log_std(self):
        return (self._log_std.exp() + self.min_std).log()

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)

        obs_flat = observation.view(T * B, -1)

        mu, cov = self.promp.mu_and_cov_y(obs_flat[:, 0])
        log_std = cov.diagonal(dim1=-2, dim2=-1).sqrt().log()
        # mu = self.mu(obs_flat)
        v = self.v(obs_flat).squeeze(-1)
        # log_std = self.log_std.repeat(T * B, 1)

        mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)
        return mu, log_std, v

    def sample_weights(self, observation=None):
        if observation is None:
            return self.promp.w_dist().sample()
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        return self.promp.w_dist().sample((T * B,))

    def compute_action(self, observation, weights):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = observation.view(T * B, -1)

        # self.sampled_weights [T*B x DN]
        B = weights.shape[0]
        phi = self.promp._get_unsqueezed_phi(obs_flat[:, 0])  # [T x 1 x 1 x M x N]
        w_blk = weights.reshape((B, 1, self.promp.D, self.promp.N, 1))  # [B X 1 x D x N x 1]
        y_blk = phi.matmul(w_blk)  # [T x 1 x D x M x 1]
        action = y_blk.reshape(B, self.promp.mD)
        return restore_leading_dims((action), lead_dim, T, B)


class AgentProMP(GaussianPgAgent):
    def __init__(self, greedy_mode, ModelCls=ModelProMP, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
        self.greedy_mode = greedy_mode
        self.sampled_weights = None

    def reset(self):
        super().reset()
        self.sampled_weights = None

    def reset_one(self, idx):
        super().reset_one(idx)
        assert self.sampled_weights is not None
        self.sampled_weights[idx, :] = self.model.sample_weights()

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )

    def step(self, observation, prev_action, prev_reward):
        if self.sampled_weights is None:
            self.sampled_weights = self.model.sample_weights(observation)

        _, agent_info = super().step(observation, prev_action, prev_reward)
        # action is not sampled from distribution but from stored weights to have smooth trajectory
        action = self.model.compute_action(observation, self.sampled_weights)

        if self.greedy_mode:
            action = agent_info.dist_info.mean
        return action, agent_info
