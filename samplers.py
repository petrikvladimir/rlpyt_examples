#!/usr/bin/env python

# Copyright (c) CTU  - All Rights Reserved
# Created on: 3/19/20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
import time

import numpy as np
from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.collections import TrajInfo
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, numpify_buffer
from rlpyt.utils.logging import logger


class BatchedEpisodicSampler(BaseSampler):

    def __init__(self, EnvCls, batch_T, batch_B, env_kwargs, TrajInfoCls=TrajInfo):
        class TmpCollector:
            mid_batch_reset = False

        self.agent = None
        self.samples_pyt = None
        self.samples_np = None
        self.agent_inputs = None
        self.env = None
        self.traj_info_kwargs = dict()
        super().__init__(EnvCls, env_kwargs, batch_T, batch_B, TmpCollector, 0, TrajInfoCls,
                         0, None, None, 0, 0)

    def initialize(self, agent, affinity=None, seed=None, bootstrap_value=False, traj_info_kwargs=None, rank=0,
                   world_size=1, ):
        assert world_size == 1  # world size used in async samplers, not relevant for this class

        T, B = self.batch_spec
        self.agent = agent
        self.env = self.EnvCls(batch_T=T, batch_B=B, **self.env_kwargs)
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(self.env.spaces, share_memory=False, global_B=B, env_ranks=env_ranks)
        self.samples_pyt, self.samples_np, examples = build_samples_buffer(agent, self.env, self.batch_spec,
                                                                           bootstrap_value, agent_shared=False,
                                                                           env_shared=False, subprocess=False,
                                                                           examples=self._get_example_outputs())

        self.samples_np.env.done[:-1, :] = False
        self.samples_np.env.done[-1, :] = True
        self.traj_info_kwargs = traj_info_kwargs

        self.agent_inputs = AgentInputs(
            buffer_from_example(examples["observation"], (B,)),
            buffer_from_example(examples["action"], (B,)),
            buffer_from_example(examples["reward"], (B,))
        )
        self._start_agent(B, env_ranks)
        logger.log("BatchedEpisodicSampler initialized.")
        return examples

    def obtain_samples(self, itr, mode='sample'):
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env

        # Reset agent inputs
        observation, action, reward = self.agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(self.agent_inputs)
        action[:], reward[:] = self.env.action_space.null_value(), 0  # reset agent inputs

        # reset environment and agent
        observation[:] = self.env.reset()
        self.agent.reset()
        agent_buf.prev_action[0], env_buf.prev_reward[0] = action, reward  # Leading prev_action.

        # perform episode
        if mode == 'sample':
            self.agent.sample_mode(itr)
        elif mode == 'eval':
            self.agent.eval_mode(itr)
        traj_infos = [self.TrajInfoCls(**self.traj_info_kwargs) for _ in range(self.batch_spec.B)]
        for t in range(self.batch_spec.T):
            env_buf.observation[t] = observation

            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)  # todo why doing this? they are sharing the same memory

            o, r, _, env_info = self.env.step(action)
            d = (t == self.batch_spec.T - 1)
            for b in range(self.batch_spec.B):
                traj_infos[b].step(observation[b], action[b], r[b], d, agent_info[b], env_info)
                if env_info:
                    env_buf.env_info[t, b] = env_info
            observation[:] = o
            reward[:] = r

            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        pass

    def _start_agent(self, B, env_ranks):
        self.agent.collector_initialize(global_B=B, env_ranks=env_ranks, )
        self.agent.reset()
        self.agent.sample_mode(itr=0)

    def _get_example_outputs(self):
        examples = dict()
        o = self.env.reset()
        a = np.stack([self.env.action_space.sample() for _ in range(self.batch_spec.B)], axis=0)
        o, r, d, env_info = self.env.step(a)

        a = np.asarray(a[0])  # get first batch only
        o = o[0]  # get first batch only
        r = np.asarray(r[0], dtype="float32")  # get first batch only, Must match torch float dtype here.
        self.agent.reset()
        agent_inputs = torchify_buffer(AgentInputs(o, a, r))
        a, agent_info = self.agent.step(*agent_inputs)
        if "prev_rnn_state" in agent_info:
            # Agent leaves B dimension in, strip it: [B,N,H] --> [N,H]
            agent_info = agent_info._replace(prev_rnn_state=agent_info.prev_rnn_state[0])
        examples["observation"] = o
        examples["reward"] = r
        examples["done"] = d
        examples["env_info"] = env_info
        examples["action"] = a  # OK to put torch tensor here, could numpify.
        examples["agent_info"] = agent_info
        return examples
