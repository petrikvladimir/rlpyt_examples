#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/19/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>


from rlpyt.agents.pg.gaussian import *
from rlpyt.agents.pg.categorical import *
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch.nn.functional as F


class ModelPgNNDiscrete(torch.nn.Module):
    def __init__(self, observation_shape, action_size,
                 policy_hidden_sizes=None, policy_hidden_nonlinearity=torch.nn.Tanh,
                 value_hidden_sizes=None, value_hidden_nonlinearity=torch.nn.Tanh, ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))

        policy_hidden_sizes = [400, 300] if policy_hidden_sizes is None else policy_hidden_sizes
        value_hidden_sizes = [400, 300] if value_hidden_sizes is None else value_hidden_sizes
        self.pi = MlpModel(input_size=input_size, hidden_sizes=policy_hidden_sizes, output_size=action_size,
                           nonlinearity=policy_hidden_nonlinearity)
        self.v = MlpModel(input_size=input_size, hidden_sizes=value_hidden_sizes, output_size=1,
                          nonlinearity=value_hidden_nonlinearity, )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = observation.view(T * B, -1)
        pi = F.softmax(self.pi(obs_flat), dim=-1)
        v = self.v(obs_flat).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


class AgentPgDiscrete(CategoricalPgAgent):
    def __init__(self, greedy_eval, ModelCls=ModelPgNNDiscrete, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
        self.greedy_eval = greedy_eval

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.n,
        )

    def step(self, observation, prev_action, prev_reward):
        action, agent_info = super().step(observation, prev_action, prev_reward)
        if self._mode == "eval" and self.greedy_eval:
            action = torch.argmax(agent_info.dist_info.prob, dim=-1)
        return action, agent_info
