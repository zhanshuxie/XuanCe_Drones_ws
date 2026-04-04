# Benchmark: Flat MARL Baselines (3D) for Hierarchical Multi-UAV Paper

## Context

用户提出了分层强化学习多无人机算法（高层 MAPPO 规划器 + 低层 PPO 控制器），需要 **4 个扁平 MARL baseline**（MAPPO、MADDPG、MASAC、MATD3）在 3D simple_spread 环境中端到端控制，与分层方法对比训练曲线。

## 需要创建/修改的文件

| 操作 | 文件 | 说明 |
|------|------|------|
| **新建** | `xuance/xuance/environment/multi_agent_env/simple_spread_drones_3d.py` | 3D 环境类 |
| **新建** | `xuance/xuance/configs/mappo/DroneSpread3D/SimpleSpreadDrones3D.yaml` | MAPPO 配置 |
| **新建** | `xuance/xuance/configs/maddpg/DroneSpread3D/SimpleSpreadDrones3D.yaml` | MADDPG 配置 |
| **新建** | `xuance/xuance/configs/masac/DroneSpread3D/SimpleSpreadDrones3D.yaml` | MASAC 配置 |
| **新建** | `xuance/xuance/configs/matd3/DroneSpread3D/SimpleSpreadDrones3D.yaml` | MATD3 配置 |
| **新建** | `xuance/examples/drones/benchmark_flat_3d.py` | Benchmark 运行脚本 |
| **修改** | `xuance/xuance/environment/multi_agent_env/__init__.py` | 注册 `DroneSpread3D` |

## Step 1: 创建 3D 环境 `simple_spread_drones_3d.py`

基于现有 2D [simple_spread_drones.py](xuance/xuance/environment/multi_agent_env/simple_spread_drones.py) 的观测结构，结合 MPE 官方的动力学和动作空间，扩展为 3D。

### 观测空间（每个 agent 21D，基于现有 2D 版扩展）
```
own_pos(3) + own_vel(3) + rel_landmarks(3×3=9) + rel_other_agents(3×2=6) = 21
```
顺序与现有 2D 版一致：`[pos, vel, rel_lm, rel_ag]`

### 动作空间（基于 MPE 官方扩展到 3D）
- `Box(0.0, 1.0, shape=(6,), dtype=float32)`
- 6D：`[−x, +x, −y, +y, −z, +z]`，每维 ∈ [0, 1]
- 净力向量：`force = [right - left, up - down, forward - backward]`
- 去掉 MPE 的 comm 维度（因为 simple_spread 中 agent 是 silent 的）

### 动力学（完全复现 MPE 官方参数）
```python
dt = 0.1          # 时间步长
damping = 0.25     # 速度衰减系数
mass = 1.0         # agent 质量
max_speed = None   # 无速度上限（同 MPE 默认）

# 每步更新顺序（与 MPE integrate_state 一致）：
p_pos = p_pos + p_vel * dt           # 1) 位置更新（用旧速度）
p_vel = p_vel * (1 - damping)        # 2) 速度衰减
p_vel = p_vel + (force / mass) * dt  # 3) 施加力
# 4) 速度裁剪（若设置了 max_speed）
# 5) 位置裁剪到 [-world_size, world_size]
```

### 全局状态（27D）
```
all_pos(3×3=9) + all_vel(3×3=9) + all_landmarks(3×3=9) = 27
```

### Reset 逻辑（与 demo_hierarchical_deploy.py 一致）
- 无人机位置：XY 随机 `rng.uniform(-ws, ws, (N, 2))`，Z 固定 1.0
- Landmark 位置：XY 随机 `rng.uniform(-ws, ws, (N, 2))`，Z 固定 1.0
- 速度：全零 `(N, 3)`

### 奖励（与 eval_hierarchical.py 的 `compute_reward_3d` 一致）
```python
reward = -Σ min_agent_dist_3d(landmark) - collision_penalty × n_collisions
```
参数：`collision_radius=0.15`, `collision_penalty=1.0`，3D 欧氏距离

### 回合参数
- `max_episode_steps` 默认 1600（= 25 × 64）
- 永不 terminated（同 MPE），仅 truncated

## Step 2: 注册环境

在 [\_\_init\_\_.py](xuance/xuance/environment/multi_agent_env/__init__.py) 的 `DroneSpread` 注册块之后添加：

```python
try:
    from xuance.environment.multi_agent_env.simple_spread_drones_3d import SimpleSpreadDrones3DEnv
    REGISTRY_MULTI_AGENT_ENV['DroneSpread3D'] = SimpleSpreadDrones3DEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["DroneSpread3D"] = str(error)
```

## Step 3: 创建 4 个算法的 YAML 配置

所有配置放在 `configs/<algo>/DroneSpread3D/SimpleSpreadDrones3D.yaml`。

### MAPPO 配置
基于现有 [SimpleSpreadDrones.yaml](xuance/xuance/configs/mappo/DroneSpread/SimpleSpreadDrones.yaml)，改动：
- `env_name: "DroneSpread3D"` / `env_id: "SimpleSpreadDrones3D"`
- `buffer_size: 1600`
- `activation_action: "sigmoid"`（匹配 [0,1] 动作范围）

### MADDPG 配置
基于 [maddpg/mpe/simple_spread_v3.yaml](xuance/xuance/configs/maddpg/mpe/simple_spread_v3.yaml)，改动：
- `env_name: "DroneSpread3D"` / `env_id: "SimpleSpreadDrones3D"`
- 其余超参数保持不变（buffer_size=100000, batch_size=256, lr_actor=0.01, lr_critic=0.001, gamma=0.95, tau=0.001, training_frequency=25）

### MASAC 配置
基于 [masac/mpe/simple_spread_v3.yaml](xuance/xuance/configs/masac/mpe/simple_spread_v3.yaml)，改动：
- `env_name: "DroneSpread3D"` / `env_id: "SimpleSpreadDrones3D"`
- 其余超参数保持不变

### MATD3 配置
基于 [matd3/mpe/simple_spread_v3.yaml](xuance/xuance/configs/matd3/mpe/simple_spread_v3.yaml)，改动：
- `env_name: "DroneSpread3D"` / `env_id: "SimpleSpreadDrones3D"`
- 其余超参数保持不变

## Step 4: 创建 Benchmark 脚本

路径：`xuance/examples/drones/benchmark_flat_3d.py`

参照 [demo_simple_spread_planner.py](xuance/examples/drones/demo_simple_spread_planner.py) 的结构，提供：
- `--algo` 参数（默认 mappo），改这个就能换算法跑 benchmark
- `--mode benchmark/train/test`
- 调用 `get_runner()` + `runner.run()`

用法：
```bash
python benchmark_flat_3d.py --algo mappo --mode benchmark
python benchmark_flat_3d.py --algo maddpg --mode benchmark
python benchmark_flat_3d.py --algo masac --mode benchmark
python benchmark_flat_3d.py --algo matd3 --mode benchmark
```

## 验证方式

```bash
cd xuance
# 跑任意一个算法的 benchmark
python examples/drones/benchmark_flat_3d.py --algo mappo --mode benchmark
# 检查 logs/<algo>/ 下的训练曲线 CSV
```
