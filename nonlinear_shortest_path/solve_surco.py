# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections import defaultdict
from pathlib import Path

import fire
import networkx as nx
import numpy as np
import pandas as pd
import pyscipopt
import torch
import tqdm.auto as tqdm
from scipy import stats
from torch import distributions

torch.manual_seed(0)

MEAN_UB = 0.2
VARIANCE_UB = 0.3

threshold_list = [("loose", 1.1), ("normal", 1), ("tight", 0.9)]


def generate_grid_instance(seed: int, n: int, mean_ub: float = MEAN_UB, variance_ub: float = VARIANCE_UB):
    """
    Generates a grid graph with values on each edge
    the means are sampled uniformly from [0, mean_ub]
    the variances are sampled uniformly from [0, variance_ub] and then multiplied by 1-mean
    """
    rng = np.random.default_rng(seed)
    graph = nx.grid_2d_graph(n, n)
    for edge in graph.edges:
        graph.edges[edge]["mean"] = rng.uniform(0.1, mean_ub)
        graph.edges[edge]["variance"] = rng.uniform(
            0.1, variance_ub) * (1 - graph.edges[edge]["mean"])
    return graph


class GridGraphDataset(object):
    """
    A dataset of grid graphs
    num_instances: number of graphs to generate
    n: number of nodes in each direction
    mean_ub: upper bound on the mean of the edge values
    variance_ub: upper bound on the variance of the edge values
    """

    def __init__(self, num_instances: int = 25, n: int = 10, mean_ub: float = MEAN_UB, variance_ub: float = VARIANCE_UB, seed: int = 0):
        self.name = f"{n}-grid"
        self.num_instances = num_instances
        self.n = n
        self.mean_ub = mean_ub
        self.variance_ub = variance_ub
        self.seed = seed
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1000, size=num_instances)
        self.graphs = []
        for s in seeds:
            G = generate_grid_instance(s, n, mean_ub, variance_ub)
            G.seed = s
            G.mean_ub = mean_ub
            G.variance_ub = variance_ub
            G.n = n
            self.graphs.append(G)

    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return self.num_instances


def get_on_time_probability(G, path, threshold):
    """
    computes the probability that the path is on time
    """
    mean = sum([G.edges[edge]["mean"] for edge in path])
    variance = sum([G.edges[edge]["variance"] for edge in path])
    return stats.norm.cdf((threshold - mean) / np.sqrt(variance))


def solve_shortest_path_with_weights(edge_weights, G, source_node, dest_node):
    edges = list(G.edges)
    torch_G = nx.to_networkx_graph(G)
    for (i, (u, v)) in enumerate(edges):
        torch_G[u][v]['weight'] = edge_weights[i]
    torch_path = nx.shortest_path(
        torch_G, source_node, dest_node, weight='weight', method="bellman-ford"
    )
    torch_path_edges = list(zip(torch_path, torch_path[1:]))
    torch_path_one_hot = torch.zeros(len(edges))
    for (u, v) in torch_path_edges:
        if (u, v) in edges:
            torch_path_one_hot[edges.index((u, v))] = 1
        else:
            torch_path_one_hot[edges.index((v, u))] = 1
    return torch_path_one_hot


class DifferentiableShortestPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_weights, G, lambda_val, source_node, dest_node):
        torch_path_one_hot = solve_shortest_path_with_weights(
            edge_weights, G, source_node=source_node, dest_node=dest_node)
        ctx.lambda_val = lambda_val
        ctx.G = G
        ctx.source_node = source_node
        ctx.dest_node = dest_node
        ctx.save_for_backward(torch_path_one_hot, edge_weights)
        return torch_path_one_hot

    @staticmethod
    def backward(ctx, grad_output):
        torch_path_one_hot, edge_weights = ctx.saved_tensors
        new_weights = edge_weights + ctx.lambda_val * grad_output
        new_weights = torch.relu(new_weights)

        improved_path = solve_shortest_path_with_weights(
            new_weights, ctx.G, source_node=ctx.source_node, dest_node=ctx.dest_node)
        grad_input = (improved_path - torch_path_one_hot)
        return grad_input, None, None, None, None


def torch_cdf(x, mu, sigma):
    return distributions.Normal(0, 1).cdf((x-mu)/sigma)


def solve_surco_zero(G, source_node, dest_node, threshold, lambda_val=1e3,
                     learning_rate=1e-1, max_iters=30, patience=5):
    """
    solves the shortest path problem using surco_zero
    """
    edges = list(G.edges)
    all_edge_data = G.edges(data=True)
    all_edge_means = torch.tensor([p["mean"] for u, v, p in all_edge_data])
    all_edge_variances = torch.tensor(
        [p["variance"] for u, v, p in all_edge_data])
    torch_raw_edge_weights = torch.randn(len(edges), requires_grad=True)
    optimizer = torch.optim.Adam([torch_raw_edge_weights], lr=learning_rate)
    history = defaultdict(list)
    best_solution = None
    best_solution_value = -np.inf
    for i in range(max_iters):
        optimizer.zero_grad()
        torch_edge_weights = torch.sigmoid(torch_raw_edge_weights)
        torch_path_one_hot = DifferentiableShortestPath.apply(
            torch_edge_weights, G, lambda_val, source_node, dest_node)
        torch_path_one_hot.retain_grad()
        path_mean = all_edge_means@torch_path_one_hot
        path_variance = all_edge_variances@torch_path_one_hot
        prob_meeting_deadline = torch_cdf(
            threshold, path_mean, torch.sqrt(path_variance))

        loss = -(threshold - path_mean)/torch.sqrt(path_variance)
        # loss = -prob_meeting_deadline

        history["cdf"].append(prob_meeting_deadline.detach().numpy().item())
        history["mean"].append(path_mean.detach().numpy().item())
        history["variance"].append(path_variance.detach().numpy().item())
        history["loss"].append(loss.detach().numpy().item())

        loss.backward()
        optimizer.step()

        if prob_meeting_deadline > best_solution_value:
            best_solution = torch_path_one_hot.detach()
            best_solution_value = prob_meeting_deadline
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter > patience:
            break
    # convert one-hot edges to list of nodes
    surco_edges = [i for i, j in zip(edges, best_solution) if j == 1]
    new_G = nx.Graph()
    new_G.add_edges_from(surco_edges)
    surco_nodes = list(nx.dfs_preorder_nodes(new_G, source_node))
    return surco_nodes, history


def solve_scip(G: nx.Graph, source_node, dest_node, threshold, timelimit):
    """
    solves the shortest path problem using scip
    G: networkx graph
    source_node: source node
    dest_node: destination node
    threshold: time threshold
    timelimit: timelimit in seconds
    """
    # solve shortest path with scip
    directed_G = G.to_directed()
    edges = directed_G.edges()
    m = pyscipopt.Model()
    # m.setIntParam("display/verblevel", 5)
    # set scip timelimit in seconds
    m.setRealParam("limits/time", timelimit)

    m.hideOutput(False)
    x = {(u, v): m.addVar(vtype="B", name=f"x_{u}_{v}")
         for (u, v) in edges}

    total_mean = sum(x[(u, v)] * directed_G[u][v]['mean'] for (u, v) in edges)
    total_variance = sum(x[(u, v)] * directed_G[u][v]
                         ['variance'] for (u, v) in edges)
    objective_var = m.addVar(vtype="C", name="objective")
    m.addCons(objective_var == (threshold - total_mean) /
              pyscipopt.sqrt(total_variance))
    m.setObjective(objective_var, "maximize")
    # incoming - outgoing == 1, -1, 0
    for i in G.nodes:
        if i == source_node:
            # one more outgoing than incoming
            rhs = 1
        elif i == dest_node:
            # 1 more incoming than outgoing
            rhs = -1
        else:
            rhs = 0
        incoming = sum(x[(u, v)] for u, v in directed_G.in_edges(i))
        outgoing = sum(x[(u, v)] for u, v in directed_G.out_edges(i))

        m.addCons(outgoing - incoming == rhs)

    m.optimize()
    solution = m.getBestSol()
    if solution is None or abs(m.getPrimalbound()) == m.infinity():
        print("couldn't find primal solution")
        return None
    solution_edges = {
        k for k, v in x.items() if m.getSolVal(solution, v) > 0.5}
    new_G = nx.Graph()
    new_G.add_edges_from(solution_edges)
    solution_nodes = list(nx.dfs_preorder_nodes(new_G, source_node))
    result = {
        "solution": solution_nodes,
        "objective": m.getSolVal(solution, objective_var),
        "gap": m.getGap(),
        "status": m.getStatus(),
        "time": m.getSolvingTime(),
        "dualbound": m.getDualbound(),
    }
    return result


def main(
    grid_n=10,
    approach="surco-zero",
    results_dir="results",
    scip_timelimit_min=30,
):
    """
    Solve the shortest path problem using a given approach

    Args:
        grid_n: size of grid graph
        approach: one of surco-zero, mean-variance, scip
        results_dir: directory to save results
        scip_timelimit_min: scip timelimit in minutes
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    test_dataset = GridGraphDataset(seed=10, n=grid_n)
    results = []
    for G in tqdm.tqdm(test_dataset, desc="graphs", total=len(test_dataset)):
        # solve LTM path problem on G with weight=mean
        source = list(G.nodes)[0]
        target = list(G.nodes)[-1]

        ltm_G = G.copy()

        ltm_path_nodes = nx.shortest_path(
            ltm_G, source, target, weight="mean")

        ltm_path_edges = list(zip(ltm_path_nodes[:-1], ltm_path_nodes[1:]))
        ltm_mean = sum([G.edges[edge]["mean"] for edge in ltm_path_edges])

        if approach == "mean-variance":
            # solve shortest path problem on G with weights=mean+variance
            mean_variance_G = G.copy()
            for u, v, d in mean_variance_G.edges(data=True):
                d["weight"] = d["mean"] + 0.5*np.sqrt(d["variance"])
            start_time = time.time()
            mean_variance_path_nodes = nx.shortest_path(
                mean_variance_G, source, target, weight="weight")
            mean_variance_time = time.time() - start_time
            mean_variance_res = {
                "approach": "mean-variance",
                "grid_n": grid_n,
                "graph_seed": G.seed,
                "graph_name": G.name,
                "source": source,
                "target": target,
                "path_solution": mean_variance_path_nodes,
                "runtime": mean_variance_time
            }

            mean_variance_path_edges = list(
                zip(mean_variance_path_nodes[:-1], mean_variance_path_nodes[1:]))

            mean_variance_mean = sum([G.edges[edge]["mean"]
                                      for edge in mean_variance_path_edges])
            mean_variance_variance = sum(
                [G.edges[edge]["variance"] for edge in mean_variance_path_edges])
            for threshold_name, threshold_mult in threshold_list:
                threshold = threshold_mult * ltm_mean
                objective_value = get_on_time_probability(
                    G, mean_variance_path_edges, threshold)
                result = mean_variance_res.copy()
                result["threshold_name"] = threshold_name
                result["threshold"] = threshold
                result["objective_value"] = objective_value
                results.append(result)
        elif approach == "surco-zero":
            # get surco solution
            for threshold_name, threshold_mult in threshold_list:
                threshold = threshold_mult * ltm_mean
                start_time = time.time()
                surco_solution, history = solve_surco_zero(
                    G, source, target, threshold)

                runtime = time.time() - start_time
                surco_res = {
                    "approach": "surco-zero",
                    "grid_n": grid_n,
                    "graph_seed": G.seed,
                    "graph_name": G.name,
                    "source": source,
                    "target": target,
                    "path_solution": surco_solution,
                    "runtime": runtime
                }
                surco_zero_path_edges = list(
                    zip(surco_solution[:-1], surco_solution[1:]))
                objective_value = get_on_time_probability(
                    G, surco_zero_path_edges, threshold)
                result = surco_res.copy()
                result["threshold_name"] = threshold_name
                result["threshold"] = threshold
                result["objective_value"] = objective_value
                results.append(result)
        elif approach == "scip":
            # get scip solution
            for threshold_name, threshold_mult in threshold_list:
                threshold = threshold_mult * ltm_mean
                start_time = time.time()
                # run scip solver for 30 minutes
                scip_result = solve_scip(
                    G, source, target, threshold, timelimit=scip_timelimit_min*60)
                if scip_result is None:
                    runtime = time.time() - start_time
                    scip_res = {
                        "approach": "scip",
                        "grid_n": grid_n,
                        "graph_seed": G.seed,
                        "graph_name": G.name,
                        "source": source,
                        "target": target,
                        "path_solution": None,
                        "runtime": runtime
                    }
                    result["threshold_name"] = threshold_name
                    result["threshold"] = threshold
                    result["objective_value"] = None
                else:
                    scip_solution = scip_result["solution"]
                    runtime = time.time() - start_time
                    scip_res = {
                        "approach": "scip",
                        "grid_n": grid_n,
                        "graph_seed": G.seed,
                        "graph_name": G.name,
                        "source": source,
                        "target": target,
                        "path_solution": scip_solution,
                        "runtime": runtime
                    }
                    surco_zero_path_edges = list(
                        zip(scip_solution[:-1], scip_solution[1:]))
                    objective_value = get_on_time_probability(
                        G, surco_zero_path_edges, threshold)
                    result = scip_res.copy()
                    result["threshold_name"] = threshold_name
                    result["threshold"] = threshold
                    result["objective_value"] = objective_value
                results.append(result)
        else:
            raise NotImplementedError(f"{approach=} is not implemented")

    result_df = pd.DataFrame(results)
    result_file = results_dir / "results.xlsx"
    if result_file.exists():
        with pd.ExcelWriter(result_file, engine='openpyxl', if_sheet_exists="overlay", mode="a") as writer:
            result_df.to_excel(
                writer, header=None, startrow=writer.sheets["Sheet1"].max_row, index=False)
    else:
        result_df.to_excel(result_file, index=False)


if __name__ == "__main__":
    """
    run this code from the nonlinear_shortest_path directory using
    python solve_surco.py --approach surco-zero
    """
    fire.Fire(main)
