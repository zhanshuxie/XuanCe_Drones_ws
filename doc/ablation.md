# 方案：分层强化学习消融实验脚本

## 背景
为分层多无人机强化学习论文（MAPPO高层规划器 + PPO低层控制器）设计消融实验，变量为 `max-low-steps`（8/16/32/64）。需要：
1. 训练两个模型，定期保存checkpoint
2. 在不同 `max-low-steps` 设置下评估所有checkpoint组合

---

## 脚本1：`train_checkpoints.py`

**用途**：分别训练MAPPO规划器和PPO导航器各10M步，每100k步保存一个checkpoint（共100个）+ 最优模型。

**运行方式**：
```bash
python train_checkpoints.py --device cuda:0 --ckpt-root checkpoints
```

### 实现思路

**阶段A - 训练MAPPO规划器**：
- 使用现有配置（`running_steps=10M`、`eval_interval=100k`、`parallels=16`）
- 循环100个epoch，每个epoch = `eval_interval // n_envs = 6250` 训练步
- 每个epoch结束后：`agent.test()` 测试 → `agent.save_model()` 保存checkpoint
- 记录最优模型

**阶段B - 训练PPO导航器**：
- 覆盖配置：`running_steps=10M`、`eval_interval=100k`（原始为3M/50k）
- 循环100个epoch，每个epoch = `100000 // 10 = 10000` 训练步
- `save_model` 会自动保存 `obs_rms.npy`（因为 `use_obsnorm=True`）
- 记录最优模型

**输出目录结构**（每个checkpoint一个子文件夹，兼容 `load_model` API）：
```
checkpoints/
  planner/
    100k/model.pth
    200k/model.pth
    ...
    10000k/model.pth
    best/model.pth
  navigator/
    100k/model.pth  + obs_rms.npy
    200k/model.pth  + obs_rms.npy
    ...
    10000k/model.pth + obs_rms.npy
    best/model.pth   + obs_rms.npy
```

> 注意：必须使用子目录结构，因为 `agent.load_model(file_path)` 会从 `os.path.dirname(file_path)` 查找 `obs_rms.npy`。

### 复用的关键文件
- [agent.py](xuance/xuance/torch/agents/base/agent.py) 的 `save_model`（第181行）—— 同时保存模型和obs_rms
- [run_marl.py](xuance/xuance/engine/run_marl.py) 的 `_run_benchmark`（第82行）—— MAPPO训练循环模板
- [run_drl.py](xuance/xuance/engine/run_drl.py) 的 `_run_benchmark`（第130行）—— PPO训练循环模板

---

## 脚本2：`eval_hierarchical.py`

**用途**：给定 `max-low-steps`，评估全部100个checkpoint组合，输出5轮平均奖励。

**运行方式**：
```bash
python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 32 --output results_mls32.csv
```

只需改 `--max-low-steps` 为 8/16/32/64 即可完成消融实验。

### 实现步骤

1. **创建agent**（仅一次）：通过 `get_runner` 创建MAPPO和PPO agent，关闭训练环境
2. **遍历100个checkpoint**（100k..10000k）：
   - `planner_agent.load_model(planner_ckpt_path)` 加载规划器
   - `nav_agent.load_model(nav_ckpt_path)` 加载导航器（自动恢复obs_rms）
3. **每个checkpoint运行5轮**：
   - 新建 `DeployAviary(gui=False)`，使用确定性随机种子
   - 初始化与原始脚本完全一致的变量：`planner_pos_2d = drone_pos_2d.copy()`、`drone_vel_2d = zeros`
   - 运行最多25个高层步骤，**每步内部逻辑与 `demo_hierarchical_deploy.py` 完全一致**：

     ```
     步骤5-a: 读取无人机实际2D位置 → drone_pos_2d[i] = state[0:2] / scale
     步骤5-b: 构建规划器观测 build_planner_obs(drone_pos_2d, drone_vel_2d, lm_pos_2d, N)
     步骤5-c: MAPPO推理 planner_agent.action([planner_obs], test_mode=True)
     步骤5-d: 更新虚拟位置和航点（与原始完全相同）:
              d2 = clip(action[:2], -1, 1) * max_step
              drone_vel_2d[i] = d2
              planner_pos_2d[i] = clip(drone_pos_2d[i] + d2, -ws, ws)
              wp3[i] = [planner_pos_2d[i,0]*scale, planner_pos_2d[i,1]*scale, height]
     步骤5-e: 低层控制器固定运行 max_low_steps 步（与原始完全相同）:
              - pending = 距航点 >= 0.05m 的无人机
              - pending无人机: obs → _process_observation → nav_agent.action
              - 已到达无人机: actions=[0,0,0,0] 悬停
              - aviary.step(actions)
     步骤5-f: 状态评估（与原始完全相同）:
              - 计算每个landmark到最近无人机的3D距离 min_dists
              - # 提前终止逻辑先注释掉（原始脚本有，但消融实验暂不启用）
              # if all(d < 0.2 for d in min_dists): break
     ```

   - **额外添加（不影响原始逻辑）**：在每个高层步骤结束后（5-f之后），在3D物理空间下计算MPE simple_spread奖励并累加
   - 关闭aviary
4. **输出CSV**：`step,avg_reward,std_reward`（100行）

### 关于提前终止
原始 `demo_hierarchical_deploy.py` 有提前终止逻辑（所有landmark被覆盖时break），但在评估脚本中**先注释掉**，固定运行25个高层步骤，便于消融实验的公平对比。

### 奖励函数（参照 [simple_spread_drones.py:187](xuance/xuance/environment/multi_agent_env/simple_spread_drones.py#L187)，改为3D距离）

所有距离计算使用3D物理坐标的二范数（`np.linalg.norm`），包括landmark覆盖距离和agent间碰撞距离：

```python
# 读取3D物理坐标
drone_pos_3d = [aviary._getDroneStateVector(i)[0:3] for i in range(N)]  # 3D
lm_xyz = ...  # 3D landmark坐标

reward = 0.0
# 1) 覆盖奖励：每个landmark到最近无人机的3D距离
for lm in lm_xyz:
    dists = [np.linalg.norm(drone_pos_3d[i] - lm) for i in range(N)]
    reward -= float(min(dists))
# 2) 碰撞惩罚：无人机之间的3D距离
for i in range(N):
    for j in range(i+1, N):
        if np.linalg.norm(drone_pos_3d[i] - drone_pos_3d[j]) < collision_radius:
            reward -= collision_penalty
```

> `collision_radius=0.15`，`collision_penalty=1.0`，与原始simple_spread一致。

### 随机种子管理
`seed = base_seed + ckpt_idx * n_episodes + ep`
- 保证可复现性
- 每轮不同初始条件
- 不同 `max-low-steps` 消融实验使用相同随机条件（公平对比）

---

## 需要创建的文件
1. `xuance/examples/drones/train_checkpoints.py` — 脚本1
2. `xuance/examples/drones/eval_hierarchical.py` — 脚本2

## 验证方法
1. 运行 `python train_checkpoints.py --device cuda:0` — 确认每个模型生成100个checkpoint
2. 运行 `python eval_hierarchical.py --max-low-steps 32 --output test.csv` — 确认生成100行的CSV
3. 检查奖励值合理性（负值，随训练进度幅度递减）
