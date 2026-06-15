



```bash
conda create -n xuance_drones_env python=3.10.20 && conda activate xuance_drones_env

```
```bash
# 安装 PyTorch 2.6+ with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
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
[details](./doc/ablation.md)

step 1: 分别训练MAPPO规划器和PPO导航器各10M步，每100k步保存一个checkpoint（共100个）+ 最优模型。

```bash
# benchmark模式
cd ./xuance/examples/drones/
python train_checkpoints.py --device cuda:0 --ckpt-root checkpoints
```

```
seed, env_seed不在parser里, 详见对应yaml文件
训练控制器:xuance/configs/ppo/Drone/NavigateAviary.yaml
训练二维规划器:xuance/configs/mappo/DroneSpread/SimpleSpreadDrones.yaml
```

step 2: 给定 max-low-steps，评估全部100个checkpoint_planner + best_naviagtor组合，输出10轮平均奖励。

```bash
# 纯测试
cd ./xuance/examples/drones/
python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 8 --output results_mls8.csv
python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 16 --output results_mls16.csv
python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 32 --output results_mls32.csv
python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 64 --output results_mls64.csv
python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 128 --output results_mls128.csv
```

```
parallals, seed在parser里
 --parallels 可指定评估时并行环境数 
 --seed 环境随机种子
```

# benchmark
[details](./doc/benchmark.md)

step 1:

```bash
# 纯测试
cd ./xuance/examples/drones/
python benchmark_hierarchical.py --ckpt-root checkpoints --max-low-steps 64 --output benchmark_hier.csv
```

```
 --parallels 可指定评估时并行环境数
 --seed 1 环境随机种子,与xuance/configs/mappo/Drones/SimpleSpreadAviary3D.yaml文件的env_seed保持一致
```

step 2:

```bash
# benchmark模式
cd ./xuance/examples/drones/
python benchmark_flat_3d.py --algo mappo --mode benchmark
python benchmark_flat_3d.py --algo maddpg --mode benchmark
python benchmark_flat_3d.py --algo masac --mode benchmark
python benchmark_flat_3d.py --algo matd3 --mode benchmark
```

```
 ---test-parallels 可指定benchmark模式评估时并行环境数 
seed, env_seed 见对应xuance/configs/xxx/Drones/SimpleSpreadAviary3D.yaml文件
```

# demo
```bash
cd ./xuance/examples/drones/
# controller
python demo_navigate.py
python demo_navigate.py --mode test --render True --model-dir results/ppo/NavigateAviary/best_model/best_model.pth --test-episode 50 --parallels 1

# 2D-planner
python demo_simple_spread_planner.py

# 看看2D planner-model + navigate_model 效果
python demo_hierarchical_deploy.py --planner-model results/mappo/SimpleSpreadDrones/best_model/best_model.pth --navigate-model results/ppo/NavigateAviary/best_model/best_model.pth --render True

# 3D-planner
python demo_simple_spread_planner_3d.py

# 看看3D planner-model + navigate_model 效果
python demo_hierarchical_deploy_3d.py --planner-model results/mappo/SimpleSpreadDrones3DPlanner/best_model/best_model.pth --navigate-model results/ppo/NavigateAviary/best_model/best_model.pth --render True
```

# 文件结构
```
xuance
├─ examples
|     └─ drones
|           ├─ benchmark_flat_3d.py     baseline基线训练脚本
|           ├─ benchmark_hierarchical.py    PPO + 二维规划器基线脚本
|           ├─ demo_hierarchical_deploy_3d.py    三维规划器部署脚本
|           ├─ demo_hierarchical_deploy.py     二维规划器部署脚本
|           ├─ demo_navigate.py     PPO控制器训练脚本
|           ├─ demo_simple_spread_planner_3d.py     三维规划器训练脚本
|           ├─ demo_simple_spread_planner.py     二维规划器训练脚本
|           ├─ eval_hierarchical.py     100个checkpoint_planner + best_naviagtor组合评估脚本
|           ├─ train_checkpoints.py     PPO / 二维规划器 100k步保存一个checkpoint训练脚本
|           ├─ visualize_hierarchical_trajectory_3d.py    三维规划器无人机轨迹可视化脚本
|           ├─ visualize_hierarchical_trajectory.py    二维规划器无人机轨迹可视化脚本
|           └─ visualize_hierarchical_trajectory_3d.py    baseline 无人机轨迹可视化脚本
└─ xuance 
    ├─ configs
    |     ├─ maddpg
    |     |    └─ Drones/SimpleSpreadAviary3D.yaml      baseline基线训练配置文件
    |     ├─ mappo
    |     |    ├─ Drones/SimpleSpreadAviary3D.yaml      baseline基线训练配置文件
    |     |    ├─ DroneSpread/SimpleSpreadDrones.yaml   二维规划器训练配置脚本
    |     |    └─ DroneSpreadPlanner3D/SimpleSpreadDrones3DPlanner.yaml   三维规划器训练配置脚本
    |     ├─ masac
    |     |    └─ Drones/SimpleSpreadAviary3D.yaml      baseline基线训练配置文件
    |     ├─ matd3
    |     |    └─ Drones/SimpleSpreadAviary3D.yaml      baseline基线训练配置文件        
    |     └─ ppo
    |         └─ Drone/NavigateAviary.yaml  PPO控制器训练配置脚本
    └─ environment
          ├─ multi_agent_env
          |      ├─ drones.py  多个无人机环境脚本
          |      ├─ simple_spread_drones_3d_planner.py  三维规划器环境脚本
          |      └─ simple_spread_drones.py  二维规划器环境脚本
          └─ single_agent_env
                 └─ drones.py  单个无人机环境脚本
```