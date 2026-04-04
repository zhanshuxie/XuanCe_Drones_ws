```bash
conda create -n xuance_drones_env python=3.10.20 && conda activate xuance_env
```

```bash
cd xuance
```

```bash
pip install -e .
```

```bash
cd gym-pybullet-drones
```

```bash
pip install -e .
```

# ablation experiments
[details](doc\ablation.md)

step 1: 分别训练MAPPO规划器和PPO导航器各10M步，每100k步保存一个checkpoint（共100个）+ 最优模型。

```bash
cd ./xuance/examples/drones/
python train_checkpoints.py --device cuda:0 --ckpt-root checkpoints
```

step 2: 给定 max-low-steps，评估全部100个checkpoint组合，输出5轮平均奖励。

```bash
cd ./xuance/examples/drones/
python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 32 --output results_mls32.csv
```

# benchmark
[details](doc\benchmark.md)

```bash
cd ./xuance/examples/drones/
python benchmark_flat_3d.py --algo mappo --mode benchmark
python benchmark_flat_3d.py --algo maddpg --mode benchmark
python benchmark_flat_3d.py --algo masac --mode benchmark
python benchmark_flat_3d.py --algo matd3 --mode benchmark
```