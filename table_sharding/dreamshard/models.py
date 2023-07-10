# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
import time
import warnings

# import pulp
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import cvxpy.settings as s
import numpy as np


class Model(nn.Module):
    def __init__(self, table_feature_dim, num_devices=4, integer_deployment=True):
        super().__init__()
        table_latent_dim = 32
        # Table feature extraction
        self.table_fc_0 = nn.Linear(table_feature_dim, 128)
        self.table_fc_1 = nn.Linear(128, table_latent_dim)

        # Table feature extraction RL
        self.rl_table_fc_0 = nn.Linear(table_feature_dim, 128)
        self.rl_table_fc_1 = nn.Linear(128, table_latent_dim)

        # Forward head
        self.forward_fc_0 = nn.Linear(table_latent_dim, 64)
        self.forward_fc_1 = nn.Linear(64, 1)

        # Communication head
        self.communication_fc_0 = nn.Linear(table_latent_dim, 64)
        self.communication_fc_1 = nn.Linear(64, 1)

        # Backward head
        self.backward_fc_0 = nn.Linear(table_latent_dim, 64)
        self.backward_fc_1 = nn.Linear(64, 1)

        # Overall head
        self.overall_fc_0 = nn.Linear(table_latent_dim, 64)
        self.overall_fc_1 = nn.Linear(64, 1)

        # Cost features extraction
        self.cost_fc_0 = nn.Linear(3, 64)
        self.cost_fc_1 = nn.Linear(64, 32)

        self.surrogate_cost_net = nn.Linear(table_latent_dim, num_devices)

        # 32 for table raw features
        # 32 for cost features
        self.policy_net = nn.Linear(32*2, 1)

        # Value net
        # 32 for table raw features
        # 32 for cost features
        # We consider 4 devices
        self.value_net = nn.Linear(32*2*4, 1)

        self.integer_deployment = integer_deployment
    
    def diff_opt_parameters(self):
        """
        _get_diff_opt_latent([[torch.stack(env.table_features)]])
        cost_estimates = self.surrogate_cost_net
        table_forward_rl,
        kernel_forward
        """
        # all params except table_fc_0, table_fc_1, overall_fc_0, overall_fc_1
        params = torch.nn.ParameterList([x for name, x in self.named_parameters() if not (name.startswith("table_fc") or name.startswith("overall_fc"))])
        # return [*self.surrogate_cost_net.parameters(), ]
        return params

    def overall_forward(self, X):
        """ Overall cost
        """
        # X: a nested list: B x number of devices x number of tables in a shard x table_feature_dim
        X = self._multi_latent(X) # B x table_latent_dim
        overall_cost = F.relu(self.overall_fc_0(X))
        overall_cost = self.overall_fc_1(overall_cost)
        overall_cost = overall_cost.flatten()

        return overall_cost

    def kernel_forward(self, X):
        """ Forward, backward, communication
        """
        # X is a list of tensors, B x number of tables in a shard x table_feature_dim
        # Forward
        X_len = torch.tensor([x.shape[0] for x in X])
        B = X_len.shape[0]

        X = torch.cat(X, dim=0)
        X = self.table_forward(X)

        ind = torch.repeat_interleave(torch.arange(len(X_len)), X_len)
        tmp = torch.zeros((X_len.shape[0], X.shape[1]))
        tmp.index_add_(0, ind, X)
        X = tmp

        # X here is batch size by latent dimension

        forward_cost = F.relu(self.forward_fc_0(X))
        forward_cost = self.forward_fc_1(forward_cost)
        forward_cost = forward_cost.flatten()

        # Communication
        communication_cost = F.relu(self.communication_fc_0(X))
        communication_cost = self.communication_fc_1(communication_cost)
        communication_cost = communication_cost.flatten()

        # Backward
        backward_cost = F.relu(self.backward_fc_0(X))
        backward_cost = self.backward_fc_1(backward_cost)
        backward_cost = backward_cost.flatten()

        return forward_cost, communication_cost, backward_cost
    
    def _get_diff_opt_latent(self, table_obs):
        X_len = torch.tensor([[x.shape[0] for x in index_X] for index_X in table_obs])
        B, D = X_len.shape
        X_len = X_len.flatten()
        table_obs = [j for sub in table_obs for j in sub]

        # Get the cost features latent
        with torch.no_grad():
            forward_cost, backward_cost, communication_cost = self.kernel_forward(table_obs)
        cost_obs = torch.cat(
            (
                forward_cost.view(B, D, -1).detach(),
                backward_cost.view(B, D, -1).detach(),
                communication_cost.view(B, D, -1).detach(),
            ),
            dim=-1,
        )
        cost_obs = F.relu(self.cost_fc_0(cost_obs))
        cost_obs = F.relu(self.cost_fc_1(cost_obs))

        # Get the table latent
        table_obs = torch.cat(table_obs, dim=0)
        table_obs = self.table_forward_rl(table_obs)
        return table_obs


    def table_forward(self, X):
        # X: B x table_feature_dim
        X = F.relu(self.table_fc_0(X))
        X = F.relu(self.table_fc_1(X))
        return X

    def table_forward_rl(self, X):
        # X: B x table_feature_dim
        X = F.relu(self.rl_table_fc_0(X))
        X = F.relu(self.rl_table_fc_1(X))
        return X

    def forward(self, obs):
        latent = self._get_latent(obs)

        # Policy head
        policy_logits = self.policy_net(latent).squeeze(-1)

        return policy_logits

    def _get_latent(self, table_obs):

        X_len = torch.tensor([[x.shape[0] for x in index_X] for index_X in table_obs])
        B, D = X_len.shape
        X_len = X_len.flatten()
        table_obs = [j for sub in table_obs for j in sub]

        # Get the cost features latent
        with torch.no_grad():
            forward_cost, backward_cost, communication_cost = self.kernel_forward(table_obs)
        cost_obs = torch.cat(
            (
                forward_cost.view(B, D, -1).detach(),
                backward_cost.view(B, D, -1).detach(),
                communication_cost.view(B, D, -1).detach(),
            ),
            dim=-1,
        )
        cost_obs = F.relu(self.cost_fc_0(cost_obs))
        cost_obs = F.relu(self.cost_fc_1(cost_obs))

        # Get the table latent
        table_obs = torch.cat(table_obs, dim=0)
        table_obs = self.table_forward_rl(table_obs)
        ind = torch.repeat_interleave(torch.arange(B*D), X_len)
        tmp = torch.zeros((B*D, table_obs.shape[1]))
        tmp.index_add_(0, ind, table_obs)
        table_obs = tmp.view(B, D, -1)

        latent = torch.cat((table_obs, cost_obs), dim=-1)
        return latent

    def _multi_latent(self, X):
        """ Get the latent for multple shards
        """
        # X: a nested list: B x number of devices x number of tables in a shard x table_feature_dim
        X_len = torch.tensor([[x.shape[0] for x in index_X] for index_X in X])
        B, D = X_len.shape
        X_len = X_len.flatten()

        X = [j for sub in X for j in sub]
        X = torch.cat(X, dim=0)
        X = self.table_forward(X)
 
        ind = torch.repeat_interleave(torch.arange(B*D), X_len)
        tmp = torch.zeros((B*D, X.shape[1]))
        tmp.index_add_(0, ind, X)
        X = tmp.view(B, D, -1)
        X = torch.max(X, dim=1)[0]

        return X    

    
    def diff_opt_forward(self, env, use_soft_solution=True, deployment=False):
        surrogate_costs = self.get_surrogate_costs(env)
        individual_table_costs = self.get_individual_table_costs(env)
        # breakpoint()
        table_sizes = env.table_sizes

        num_tables = len(env.table_features)
        num_devices = env.ndevices
        max_memory = env.max_memory

        x = cp.Variable(
            (num_tables, num_devices),
            name="table_device_assignment",
            # nonneg=True,
            integer=True if deployment else False # explicitly say integer in deployment
            )
        z = cp.Variable(name="obj")
        costs = cp.Parameter((num_tables, num_devices), nonneg=True)
        individual_costs = cp.Parameter((num_tables), nonneg=True)
        constraints = [
            x >= 0, # who knows if nonneg works
            x <= 1,
            cp.sum(x, axis=1) == 1,
            cp.sum(x, axis=0) >= 1,
            np.array(table_sizes) @ x <= max_memory,
            # individual_table_costs @ x <= z
            ]
        # max latency objective
        # objective = cp.Minimize(cp.sum(cp.multiply(costs, x)) + cp.max(individual_costs @ x))
        objective = cp.Minimize(cp.sum(cp.multiply(costs, x)))
        # squared latency objective
        # objective = cp.Minimize(cp.sum(cp.multiply(costs, x)) + cp.sum(cp.power(individual_costs @ x, 2)))
        problem = cp.Problem(objective, constraints)

        cvxpylayer = CvxpyLayer(problem, parameters=[costs, individual_costs], variables=[x])

        # solve the problem
        solution, = cvxpylayer(surrogate_costs, individual_table_costs)

        # correct some small infeasibilities in assignment
        soft_assignment = F.relu(solution)
        soft_assignment = soft_assignment / soft_assignment.sum(axis=1)[:, None]
        if use_soft_solution:
            return soft_assignment
        else:
            hard_assignment = F.one_hot(torch.argmax(soft_assignment, axis=1), num_classes=num_devices)
            # use pass through gradient
            return soft_assignment + (hard_assignment - soft_assignment).detach()
            # if samples are needed then use this
            # hard_assignment = torch.multinomial(soft_assignments, 1)    

    def get_surrogate_costs(self, env):
        table_latent_obs = self._get_diff_opt_latent([[torch.stack(env.table_features)]])
        cost_estimates = self.surrogate_cost_net(table_latent_obs)
        cost_estimates = torch.sigmoid(cost_estimates)
        return cost_estimates
    
    def get_individual_table_costs(self, env):
        # returns list of table costs
        all_costs = []
        for f in env.table_features:
            forward_cost, backward_cost, communication_cost = self.kernel_forward([f.unsqueeze(0)])
            all_costs.append(forward_cost + backward_cost + communication_cost)
        all_costs = torch.cat(all_costs)
        return all_costs
    
    def on_the_fly_opt_forward(self, env, deployment=False, warm_start_surrogate_costs=False, use_soft_solution=False):
        
        # self.get_surrogate_costs(env)
        individual_table_costs = self.get_individual_table_costs(env)
        table_sizes = env.table_sizes

        num_tables = len(env.table_features)
        num_devices = env.ndevices
        max_memory = env.max_memory
        if warm_start_surrogate_costs:
            surrogate_costs = self.get_surrogate_costs(env).clone().detach().requires_grad_(True)
        else:
            surrogate_costs = torch.rand((num_tables, num_devices), requires_grad=True) # constant updateable tensor

        x = cp.Variable(
            (num_tables, num_devices),
            name="table_device_assignment",
            # nonneg=True,
            integer=True if deployment else False # explicitly say integer in deployment
            )
        z = cp.Variable(name="obj")
        costs = cp.Parameter((num_tables, num_devices), nonneg=True)
        individual_costs = cp.Parameter((num_tables), nonneg=True)
        constraints = [
            x >= 0, # who knows if nonneg works
            x <= 1,
            cp.sum(x, axis=1) == 1,
            cp.sum(x, axis=0) >= 1,
            np.array(table_sizes) @ x <= max_memory,
            # individual_table_costs @ x <= z
            ]
        # max latency objective
        # objective = cp.Minimize(cp.sum(cp.multiply(costs, x)) + cp.max(individual_costs @ x))
        objective = cp.Minimize(cp.sum(cp.multiply(costs, x)))
        # squared latency objective
        # objective = cp.Minimize(
        #     cp.sum(cp.multiply(costs, x)) + cp.sum(cp.power(individual_costs @ x, 2))
        #     )
        problem = cp.Problem(objective, constraints)

        cvxpylayer = CvxpyLayer(problem, parameters=[costs, individual_costs], variables=[x])
        optimizer = torch.optim.Adam(
            [surrogate_costs],
            lr=0.05,
        )
        for i in range(20):
            optimizer.zero_grad()
            # iteratively train the model to produce better solutions
            # solve the problem
            solution, = cvxpylayer(surrogate_costs, individual_table_costs)

            # correct some small infeasibilities in assignment
            soft_assignment = F.relu(solution)
            soft_assignment = soft_assignment / soft_assignment.sum(axis=1)[:, None]
            hard_assignment = F.one_hot(torch.argmax(soft_assignment, axis=1), num_classes=num_devices)
            solution = hard_assignment
            loss = env.model.evaluate_solution(solution, env)
            loss.backward()
            # print(loss)
            optimizer.step()
            # if samples are needed then use this
            # hard_assignment = torch.multinomial(soft_assignments, 1)    

        
        costs = surrogate_costs.detach().numpy()
        individual_table_costs = self.get_individual_table_costs(env).detach().numpy()
        table_sizes = env.table_sizes

        num_tables = len(env.table_features)
        num_devices = env.ndevices
        max_memory = env.max_memory

        x = cp.Variable(
            (num_tables, num_devices),
            name="table_device_assignment",
            # nonneg=True,
            integer=True,
            # integer=False,
            )
        # costs = cp.Parameter((num_tables, num_devices), nonneg=True)
        # individual_costs = cp.Parameter((num_tables), nonneg=True)
        individual_costs = individual_table_costs
        constraints = [
            x >= 0, # who knows if nonneg works
            x <= 1,
            cp.sum(x, axis=1) == 1,
            cp.sum(x, axis=0) >= 1,
            np.array(table_sizes) @ x <= max_memory,
            # individual_table_costs @ x <= z
            ]
        # max latency objective
        # objective = cp.Minimize(cp.sum(cp.multiply(costs, x)) + cp.max(individual_costs @ x))
        objective = cp.Minimize(cp.sum(cp.multiply(costs, x)))
        # squared latency objective
        # objective = cp.Minimize(
        #     cp.sum(cp.multiply(costs, x))
        #     # + cp.sum(cp.power(individual_costs @ x, 2))
        #     )
        problem = cp.Problem(objective, constraints)
        if self.integer_deployment:
            problem.solve(solver=MySCIP())
        else:
            problem.solve()
        
        solution = x.value
        soft_assignment = np.maximum(solution, 0)
        soft_assignment = soft_assignment / soft_assignment.sum(axis=1)[:, None]
        hard_assignment = np.zeros((num_tables, num_devices))
        hard_assignment[np.arange(num_tables), np.argmax(soft_assignment, axis=1)] = 1
        return hard_assignment


    def opt_forward(self, env):
        
        surrogate_costs = self.get_surrogate_costs(env).detach().numpy()
        individual_table_costs = self.get_individual_table_costs(env).detach().numpy()
        table_sizes = env.table_sizes

        num_tables = len(env.table_features)
        num_devices = env.ndevices
        max_memory = env.max_memory

        x = cp.Variable(
            (num_tables, num_devices),
            name="table_device_assignment",
            integer=True,
            )
        # costs = cp.Parameter((num_tables, num_devices), nonneg=True)
        costs = surrogate_costs
        # individual_costs = cp.Parameter((num_tables), nonneg=True)
        individual_costs = individual_table_costs
        constraints = [
            x >= 0, # who knows if nonneg works (it doesn't)
            x <= 1,
            cp.sum(x, axis=1) == 1,
            cp.sum(x, axis=0) >= 1,
            np.array(table_sizes) @ x <= max_memory,
            # individual_table_costs @ x <= z
            ]
        # max latency objective
        # objective = cp.Minimize(cp.sum(cp.multiply(costs, x)) + cp.max(individual_costs @ x))
        # squared latency objective
        objective = cp.Minimize(cp.sum(cp.multiply(costs, x)) + cp.sum(cp.power(individual_costs @ x, 2)))
        problem = cp.Problem(objective, constraints)
        if self.integer_deployment:
            problem.solve(solver=MySCIP())
        else:
            problem.solve()
        
        solution = x.value
        soft_assignment = np.maximum(solution, 0)
        soft_assignment = soft_assignment / soft_assignment.sum(axis=1)[:, None]
        hard_assignment = np.zeros((num_tables, num_devices))
        hard_assignment[np.arange(num_tables), np.argmax(soft_assignment, axis=1)] = 1
        return hard_assignment

    
    # need to call kernel_forward
    def evaluate_solution(self, hard_solution, env):
        # soft_solution: num_tables x num_devices
        # for each device build nested list of solutions
        num_tables = len(env.table_features)
        num_devices = env.ndevices
        obs = []
        for device_ind in range(num_devices):
            obs.append([])
        for table_ind in range(num_tables):
            selected_device = hard_solution[table_ind].argmax()
            obs[selected_device].append(hard_solution[table_ind, selected_device] * env.table_features[table_ind])
        for device_ind in range(num_devices):
            if obs[device_ind] == []:
                obs[device_ind] = torch.zeros((0, env.num_features))
            else:
                obs[device_ind] = torch.stack(obs[device_ind])
        # kernel_forward = self.kernel_forward([obs])
        predicted_quality = self.overall_forward([obs])
        return predicted_quality


class MySCIP(cp.reductions.solvers.conic_solvers.scip_conif.SCIP):
    
    def name(self):
        return "CUSTOM_SCIP"
    
    def _set_params(
        self,
        model,
        verbose: bool,
        solver_opts,
        data,
        dims,
    ) -> None:
        """Set model solve parameters."""
        from pyscipopt import SCIP_PARAMSETTING

        # Set model verbosity
        hide_output = not verbose
        model.hideOutput(hide_output)

        # General kwarg params
        scip_params = solver_opts.pop("scip_params", {})
        if solver_opts:
            try:
                model.setParams(solver_opts)
            except KeyError as e:
                raise KeyError(
                    "One or more solver params in {} are not valid: {}".format(
                        list(solver_opts.keys()),
                        e,
                    )
                )

        # Scip specific params
        if scip_params:
            try:
                model.setParams(scip_params)
            except KeyError as e:
                raise KeyError(
                    "One or more scip params in {} are not valid: {}".format(
                        list(scip_params.keys()),
                        e,
                    )
                )

        is_mip = data[s.BOOL_IDX] or data[s.INT_IDX]
        has_soc_constr = len(dims[s.SOC_DIM]) > 1
        if not (is_mip or has_soc_constr):
            # These settings are needed  to allow the dual to be calculated
            model.setPresolve(SCIP_PARAMSETTING.OFF)
            model.setHeuristics(SCIP_PARAMSETTING.OFF)
            model.disablePropagation()
