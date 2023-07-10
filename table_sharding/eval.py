# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
import os
import time
import traceback
from pathlib import Path

import faker
import numpy as np
import pandas as pd
import torch

from dreamshard import sharders
from dreamshard.env import Env
from dreamshard.models import Model
from dreamshard.multi_gpu_bench_interface import Evaluator
from dreamshard.utils import (allocation2plan, get_table_ids_list,
                              load_table_configs_features_sizes, table_size)


def main():
    parser = argparse.ArgumentParser("Benchmark sharding")
    parser.add_argument('--data-dir', type=str, default="processed_data/dlrm_datasets/fbgemm_t856_bs65536")
    parser.add_argument('--task-path', type=str, default="processed_data/dlrm_tasks_50/test.txt")
    parser.add_argument('--alg', type=str, default="random", choices=["naive", "dim_greedy", "size_greedy", "lookup_greedy", "size_lookup_greedy", "random", "dreamshard", "surco_prior", "surco_zero", "surco_hybrid"])
    parser.add_argument('--model', type=str, default=None, help="model file for algorithms that require a model")
    parser.add_argument('--max-memory', type=int, default=5, help="Max memory for each shard in GB")
    parser.add_argument('--gpu-devices', type=str, default="0,1,2,3")
    parser.add_argument('--integer-deployment', type=bool, default=False)

    args = parser.parse_args()
    args.ndevices = len(args.gpu_devices.split(","))

    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)

    model = Model(table_feature_dim=21, num_devices=args.ndevices, integer_deployment=args.integer_deployment)
    if args.model:
        model.load_state_dict(torch.load(args.model))
    # if args.alg[-3:] == ".pt":
    #     # args.alg = "dreamshard"
    #     # args.alg = "diffopt"
    #     args.alg = "onthefly_diffopt"
    
    table_ids_list = get_table_ids_list(args.task_path)
    print(f"solving {len(table_ids_list)} problems")
    try:
        evaluator = Evaluator(
            args.data_dir,
            args.task_path,
            args.gpu_devices,
        )
        latencies = [] 
        runtimes = []
        for task_id, table_ids in enumerate(table_ids_list):
            print("Task", str(task_id+1)+"/"+str(len(table_ids_list)))
            start_time = time.time()
            table_configs, table_features, table_sizes = load_table_configs_features_sizes(args.data_dir, table_ids)
            
            env = Env(
                table_features,
                table_sizes,
                model,
                args.ndevices,
                args.max_memory,
            )
            env.table_configs = table_configs

            sharding = sharders.shard(env, args.alg)
            print("Sharding:", sharding)
            plan = allocation2plan(sharding, env.ndevices)
            # Dim sums
            dims = [config["dim"] for config in env.table_configs]
            dim_sums = [sum([dims[i] for i in shard]) for shard in plan]
            print("Dims:", dim_sums)

            # Check size
            sizes = [table_size(config["row"], config["dim"], fp16=True) for config in env.table_configs]
            size_sums = [sum([sizes[i] for i in shard]) for shard in plan]
            print("Sizes:", size_sums)
            max_size_sum = max(size_sums)
            if max_size_sum > env.max_memory: 
                print("Out of memory")
                continue

            max_latency, latency = evaluator.evaluate(task_id, sharding)
            latencies.append(max_latency)
            print("Latency:", max_latency)
            runtime = time.time() - start_time
            runtimes.append(runtime)
        
        print("latencies")
        print("Average:", np.mean(latencies))
        print("stdev:", np.std(latencies)/np.sqrt(len(latencies)))
        print(f"{np.mean(latencies):.1f}$\pm${np.std(latencies)/np.sqrt(len(latencies)):.1f}")
        print(latencies)

        print("runtimes")
        print("Average:", np.mean(runtimes))
        print("stdev:", np.std(runtimes)/np.sqrt(len(runtimes)))
        print(f"{np.mean(runtimes):.1f}$\pm${np.std(runtimes)/np.sqrt(len(runtimes)):.3f}")
        print(runtimes)

        # save latencies and runtimes
        # get unique identifier for this run
        run_id_generator = faker.Faker()
        run_id = "-".join(run_id_generator.words(unique=True))
        # save to a file
        result_df = pd.DataFrame({
            "latency": latencies,
            "runtime": runtimes,
        })
        result_df["alg"] = args.alg
        result_df["ndevices"] = args.ndevices
        result_df["max_memory"] = args.max_memory
        result_df["run_id"] = run_id
        result_df["model"] = args.model


        result_df.to_csv(
            result_dir / f"{args.alg}_{run_id}.csv",
            index=False,
        )

    except:
        traceback.print_exc()
    finally:
        evaluator.terminate()


if __name__ == '__main__':
    main()

