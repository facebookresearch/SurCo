# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch


def gen_table_configs(
    lS_rows,
    lS_pooling_factors,
    lS_bin_counts,
    args
):
    T = len(lS_rows)  # number of tables

    table_configs = {}
    table_configs["tables"] = []
    for i in range(T):
        table_config = {}
        table_config["index"] = i
        table_config["row"] = int(lS_rows[i])
        table_config["dim"] = int(args.dim)
        table_config["pooling_factor"] = int(lS_pooling_factors[i])
        for j, bin_count in enumerate(lS_bin_counts[i]):
            table_config["bin_"+str(j)] = lS_bin_counts[i][j]

        table_configs["tables"].append(table_config)
    return table_configs


def process_data(data_path):
    indices, offsets, lengths = torch.load(data_path)
    num_tables, batch_size = lengths.shape

    indices = indices.cuda()
    offsets = offsets.cuda()

    lS_pooling_factors = list(map(int, lengths.float().mean(dim=1).tolist()))

    # Split the tables
    lS_offsets, lS_indices, lS_rows, lS_bin_counts = [], [], [], []
    for t in range(num_tables):
        start = t * batch_size
        end = (t + 1) * batch_size + 1
        table_offsets = offsets[start:end]
        table_indices = indices[table_offsets[0]:table_offsets[-1]]
        table_offsets = table_offsets - offsets[start]

        row = table_indices.max().int().item() + 1 if len(table_indices) > 0 else 1
        row = max(100, row)

        _, indices_counts = torch.unique(table_indices, return_counts=True)
        unique_counts, counts_counts = torch.unique(
            indices_counts, return_counts=True)
        total_counts = counts_counts.sum().item()

        if total_counts == 0:
            bin_counts = [0.0 for _ in range(17)]
        else:
            start, end = 0, 1
            bin_counts = []
            for i in range(16):
                bin_counts.append(counts_counts[(unique_counts > start) & (
                    unique_counts <= end)].sum().item())
                start = end
                end *= 2
            bin_counts.append(
                counts_counts[unique_counts > start].sum().item())
            bin_counts = [x/total_counts for x in bin_counts]

        lS_offsets.append(table_offsets.detach().clone().cpu())
        lS_indices.append(table_indices.detach().clone().cpu())
        lS_rows.append(row)
        lS_bin_counts.append(bin_counts)

    return lS_offsets, lS_indices, lS_rows, lS_pooling_factors, lS_bin_counts


def main():
    parser = argparse.ArgumentParser("Process DLRM data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--data', type=str, default="dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536.pt")
    parser.add_argument('--out-dir', type=str, default="processed_data/dlrm_datasets/fbgemm_t856_bs65536")
    parser.add_argument('--dim', type=int, default=16)

    args = parser.parse_args()
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    data_file = Path(args.data)
    assert data_file.exists(), f"Data file {args.data} does not exist"
    
    print("Procssing DLRM data...")
    lS_offsets, lS_indices, lS_rows, lS_pooling_factors, lS_bin_counts = process_data(
        args.data)
    data = {
        "lS_offsets": lS_offsets,
        "lS_indices": lS_indices,
    }

    torch.save(data, out_dir / f"data.pt")

    print("Generating table configs...")
    table_configs = gen_table_configs(
        lS_rows,
        lS_pooling_factors,
        lS_bin_counts,
        args
    )
    with open(out_dir / f"table_configs.json", "w") as f:
        json.dump(table_configs, f)


if __name__ == '__main__':
    """
    Usage:
        python tools/gen_dlrm_data.py --data dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536.pt --out-dir processed_data/dlrm_datasets/fbgemm_t856_bs65536
    """
    main()
