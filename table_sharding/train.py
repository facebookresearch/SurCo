# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse

import numpy as np
import torch

from dreamshard.training import train

def main():
    parser = argparse.ArgumentParser("RLShard")
    parser.add_argument('--data-dir', type=str, default="processed_data/dlrm_datasets/fbgemm_t856_bs65536/")
    parser.add_argument('--task-path', type=str, default="processed_data/dlrm_tasks_50/fbgemm_t856_bs65536/train.txt")
    parser.add_argument('--gpu-devices', type=str, default="1,2,3,4")
    parser.add_argument('--num-iterations', type=int, default=10)
    parser.add_argument('--bench-steps', type=int, default=10)
    parser.add_argument('--rl-num-batches', type=int, default=10)
    parser.add_argument('--rl-batch-size', type=int, default=20)
    parser.add_argument('--bench-training-steps', type=int, default=300)
    parser.add_argument('--entropy-weight', type=int, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-memory', type=int, default=5, help="Max memory for each shard in GB")
    parser.add_argument('--out-dir', type=str, default="models/dreamshard")
    # set options to be rl, surco-zero, surco-prior, surco-hybrid

    parser.add_argument('--alg', type=str, default="rl", choices=["rl", "surco"])
    parser.add_argument('--integer-deployment', type=bool, default=False)
    args = parser.parse_args()
    args.ndevices = len(args.gpu_devices.split(","))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    train(args)

if __name__ == '__main__':
    """
    python 
    """
    main()
