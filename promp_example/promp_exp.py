#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-16
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import torch
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.logger import record_tabular_misc_stat
from rlpyt_utils import args
from rlpyt_utils.agents_nn import AgentPgContinuous
from rlpyt_utils.promp.promp import ProMP
from rlpyt_utils.runners.minibatch_rl import MinibatchRlWithLog

from promp_example.promp_envs import EnvProMP
from promp_example.agents_promp import ModelProMP, AgentProMP
import numpy as np

parser = args.get_default_rl_parser()
args.add_default_ppo_args(parser)
options = parser.parse_args()

horizon = 50
sampler = SerialSampler(
    EnvCls=EnvProMP,
    env_kwargs=dict(horizon=horizon, weight_are_actions=True),
    batch_T=horizon if args.is_evaluation(options) else horizon * 8,
    batch_B=32 if args.is_evaluation(options) else 8,
    max_decorrelation_steps=0,
)

init_agent = args.load_initial_model_state(options)
# init_agent['promp.cov_w_params'] *=0 + 0.1
agent = AgentProMP(
    options.greedy_eval,
    initial_model_state_dict=init_agent,
    model_kwargs=dict(
        value_hidden_sizes=[64, 64], value_hidden_nonlinearity=torch.nn.Tanh,
        policy_hidden_sizes=[64, 64], policy_hidden_nonlinearity=torch.nn.Tanh,
        init_log_std=np.log(0.1),
        # policy_input_dims=[2, 3]
    ),
)


#
# agent = AgentPgContinuous(
#     options.greedy_eval,
#     # ModelCls=DMPUnitParams,
#     initial_model_state_dict=args.load_initial_model_state(options),
#     model_kwargs=dict(
#         value_hidden_sizes=[64, 64], value_hidden_nonlinearity=torch.nn.Tanh,
#         policy_hidden_sizes=[64, 64], policy_hidden_nonlinearity=torch.nn.Tanh,
#         init_log_std=np.log(0.1),
#         # policy_input_dims=[2, 3]
#     ),
# )

def log_diagnostics(itr, algo, agent, sampler):
    mp: ProMP = agent.model.promp
    mu, cov = mp.mu_and_cov_w
    std = cov.diagonal(dim1=-2, dim2=-1).sqrt().detach().numpy()
    # for i in range(std.shape[0]):
    #     record_tabular('agent/std{}'.format(i), std[i])
    record_tabular_misc_stat('AgentCov', std)
    record_tabular_misc_stat('AgentMu', mu.detach().numpy())


runner = MinibatchRlWithLog(algo=args.get_ppo_from_options(options),
                            agent=agent, sampler=sampler, log_traj_window=32,
                            n_steps=int(500000 * sampler.batch_size),
                            log_interval_steps=int(horizon * 32),
                            log_diagnostics_fun=log_diagnostics
                            )

if not args.is_evaluation(options):
    with args.get_default_context(options):
        runner.train()
else:
    runner.startup()
    while True:
        sampler.obtain_samples(0)
        print(np.sum(sampler.samples_np.env.reward, 0))
        EnvProMP.plot_states(sampler.samples_np.env.observation, sampler.samples_np.agent.action, max_obs=64)
