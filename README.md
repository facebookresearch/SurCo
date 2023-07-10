# SurCo

## Installation instructions
Each setting has its own installation instructions and requirements.

### Clone the repository

```bash
git clone 
cd surco
```

### Install dependencies

```bash
conda env create -n surco -f environment.yml
conda activate surco
```

**For Table Sharding: install FBGEMM**

Follow the instructions in [https://github.com/pytorch/FBGEMM](https://github.com/pytorch/FBGEMM) to install the embedding operators

If you are using a100 or v100 gpus you might be able to install with pip
```bash
pip install fbgemm_gpu
```

## Run experiments


### **Nonlinear Shortest Path**
Run the experiments from the `surco/nonlinear_shortest_path` directory.
```bash
cd nonlinear_shortest_path
python -m solve_surco.py --approach surco-zero
python -m solve_surco.py --approach mean-variance
python -m solve_surco.py --approach scip
```
Results are written to `surco/nonlinear_shortest_path/results`.


### **Table Sharding**

Run experiments from the `table_sharding` directory
```bash
cd table_sharding
```

**Step 1: Download DLRM dataset**

Download the data with [`git lfs`](https://github.com/git-lfs/git-lfs) at [https://github.com/facebookresearch/dlrm_datasets](https://github.com/facebookresearch/dlrm_datasets).
If needed, install `git lfs` following instructions here: [https://github.com/git-lfs/git-lfs#installing](https://github.com/git-lfs/git-lfs#installing). You can most likely install using conda
``` bash
conda install -c conda-forge git-lfs
git lfs install
```
Clone the repository and download the data
``` bash
git lfs clone git@github.com:facebookresearch/dlrm_datasets.git
```

Decompress data files, e.g. on linux
``` bash
gunzip dlrm_datasets/embedding_bag/*/*.pt.gz
``` 

**Step 2: Process the dataset**
``` bash
(EMBEDDING=fbgemm_t856_bs65536 && python tools/gen_dlrm_data.py --data dlrm_datasets/embedding_bag/2021/$EMBEDDING.pt --out-dir processed_data/dlrm_datasets/$EMBEDDING)
```
Note that you need to change `--data` argument to the path where you downloaded DLRM dataset.

**Step 3: Generate training and testing tasks**
```bash
(EMBEDDING=fbgemm_t856_bs65536 && NUM_TABLES=50 && python tools/gen_tasks.py --T $NUM_TABLES --data-dir=processed_data/dlrm_datasets/$EMBEDDING --out-dir processed_data/dlrm_tasks_$NUM_TABLES/$EMBEDDING)
```
The argument `--T` specifies the number of tables, and `--out-dir` indicates the output directory.

Note, you can loop from 10 to 60 to create a couple of instances using
```bash
for NUM_TABLES in {10..60..10}
do
    (EMBEDDING=fbgemm_t856_bs65536 && python tools/gen_tasks.py --T $NUM_TABLES --data-dir=processed_data/dlrm_datasets/$EMBEDDING --out-dir processed_data/dlrm_tasks_$NUM_TABLES/$EMBEDDING)
done
```

**Step 4: Train DreamShard**
```bash
(EMBEDDING=fbgemm_t856_bs65536 && NUM_TABLES=50 && python train.py --task-path processed_data/dlrm_tasks_$NUM_TABLES/$EMBEDDING/train.txt --gpu-devices 0,1,2,3 --max-memory 5 --out-dir models/dreamshard_$NUM_TABLES/$EMBEDDING)
```
Note that you need to specify `--gpu-devices` and `--max-memory` based on your GPU setup. You also need to specify `--task-path`. `--out-dir` indicates where the trained model will be saved.

**Step 5: Evaluate DreamShard and baselines**
```bash
(EMBEDDING=fbgemm_t856_bs65536 && NUM_TABLES=50 && python eval.py --task-path processed_data/dlrm_tasks_$NUM_TABLES/$EMBEDDING/test.txt --gpu-devices 0,1,2,3 --max-memory 5 --alg=dreamshard --model=models/dreamshard_$NUM_TABLES/$EMBEDDING/rl_9.pt)
```
Not that you need to specify `--gpu-devices` and `--max-memory` based on ***your GPU***. You also need to specify `--task-path`. `--alg` points to the saved model. Here `9.pt` is the final saved model because we train 10 iterations and save the model after each iteration.

To obtain the results of the baselines, simply change `--alg` to `random`, `dim_greedy`, `lookup_greedy`, `size_greedy`, or `size_lookup_greedy`.
<!-- 
Run the experiments from the `surco/table_sharding` directory.
```bash
cd table_sharding
python -m solve_surco.py --approach surco-zero
python -m solve_surco.py --approach mean-variance
python -m solve_surco.py --approach scip
``` -->
<!-- Results are written to `surco/nonlinear_shortest_path/results`. -->



### Inverse Photonics
Run the experiments from the `surco/table_sharding` directory.
```bash
cd table_sharding
python -m solve_surco.py --approach surco-zero
python -m solve_surco.py --approach mean-variance
python -m solve_surco.py --approach scip
```
Results are written to `surco/nonlinear_shortest_path/results`.

## LICENSE
The project is under CC BY-NC 4.0. Please check LICENSE file for details.
