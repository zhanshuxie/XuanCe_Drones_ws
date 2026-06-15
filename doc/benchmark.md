# 计划：在 multi_agent_env/drones.py 中添加 SimpleSpreadAviary3D

## 背景

用户需要一个基于 PyBullet 物理引擎的多智能体 3D SimpleSpread 环境，用来**替代**原有的 `simple_spread_drones_3d.py`（MPE 简化物理）作为 benchmark。结合：
- **NavigateAviary 的物理引擎和观测/动作格式**：PyBullet 真实无人机动力学 + PID 速度控制器 + KIN 运动学观测
- **SimpleSpreadDrones3DEnv 的任务逻辑**：合作覆盖 landmark 的奖励函数、无空间约束、仅基于步数的回合截断、随机化 reset
- `max_episode_steps = 1600`（与原 SimpleSpreadDrones3DEnv 一致）
- 无人机运动空间**不加约束**（不因出界/倾斜截断）

目标：用户运行 `python benchmark_flat_3d.py --env Drones --env-id SimpleSpreadAviary3D --algo mappo` 即可训练

## 需要修改的文件

### 1. `xuance/xuance/environment/multi_agent_env/drones.py` — 主要修改

**1a. 添加 import**（第 13 行，try 块内）：
```python
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
import pybullet as p
```

**1b. 新增类 `SimpleSpreadAviary3D(BaseRLAviary)`** — 插入到 REGISTRY 之前（约第 178 行）：

构造函数 `__init__`：
- 参数：`num_drones=3, num_landmarks=3, world_size=2.0, collision_radius=0.15, collision_penalty=1.0, flight_height=1.0, act=ActionType.VEL, obs=ObservationType.KIN, ctrl_freq=30, pyb_freq=240, ...`
- `EPISODE_LEN_SEC = 60`（设得比 1600/30≈53.3秒 更长，让 wrapper 的 max_episode_steps=1600 来控制截断）
- `SPEED_LIMIT = 0.3` m/s（BaseRLAviary 在 VEL 模式下自动设置为 `0.03 * MAX_SPEED_KMH * 1000/3600 = 0.3`，无需额外配置）
- **必须在 `super().__init__()` 之前** 初始化 `self.NUM_LANDMARKS`、`self.landmark_pos`、`self.world_size` 等，因为父类构造函数会调用 `_addObstacles()`

`_addObstacles(self)`：
- 随机化无人机位置：XY 在 `[-world_size, world_size]`，Z = `flight_height`，通过 `p.resetBasePositionAndOrientation()` + `p.resetBaseVelocity()` 设置每架无人机
- 随机化 landmark 位置：XY 在 `[-world_size, world_size]`，Z = `flight_height`
- 如果 GUI 模式：用 `p.createVisualShape` + `p.createMultiBody` 为每个 landmark 创建可视化球体

`_observationSpace(self)`：
- 每架无人机观测维度：9（rpy + rpy_rate + vel）+ 3×num_landmarks（相对 landmark 位置）+ 3×(num_drones-1)（相对其他 agent 位置）+ 4×ACTION_BUFFER_SIZE（动作历史）
- 3 agents + 3 landmarks 时：9 + 9 + 6 + 15 * 4 = **84 维/无人机**
- 返回 `Box(shape=(NUM_DRONES, obs_dim))`

`_computeObs(self)`：
- 对每架无人机 i：`state = _getDroneStateVector(i)`
  - `drone_state = [state[7:10], state[10:13], state[13:16]]`（rpy, rpy_rate, vel — 与 NavigateAviary 一致）
  - `rel_landmarks = (landmark_pos - state[0:3])` 对每个 landmark
  - `rel_agents = (other_drone_pos - state[0:3])` 对每个其他无人机
- 追加 action buffer（与 NavigateAviary 模式一致）
- 返回形状 `(NUM_DRONES, obs_dim)`

`_computeReward(self)`：
- 覆盖奖励：`-Σ min_agent_dist(landmark)`（与 SimpleSpreadDrones3DEnv 一致）
- 碰撞惩罚：每对距离 < `collision_radius` 的 agent 减去 `collision_penalty`
- **共享合作奖励**，返回形状 `(NUM_DRONES, 1)`（wrapper 通过 `reward[i, 0]` 索引）

`_computeTerminated(self)`：始终返回 `False`（与 SimpleSpreadDrones3DEnv 一致）

`_computeTruncated(self)`：`step_counter / PYB_FREQ > EPISODE_LEN_SEC`（仅时间截断，**不检查出界/倾斜**，无人机运动空间无约束）

`_computeInfo(self)`：`{"episode_step": self.step_counter}`

**1c. 更新 REGISTRY**（第 178 行）：
```python
REGISTRY = {
    "MultiHoverAviary": MultiHoverAviary,
    "SimpleSpreadAviary3D": SimpleSpreadAviary3D,
}
```

**1d. 更新 `Drones_MultiAgentEnv` wrapper**：

- 第 194 行：扩展 env_id 判断条件，包含 `"SimpleSpreadAviary3D"`
- 为 SimpleSpreadAviary3D 传递额外参数：`num_landmarks, world_size, collision_radius, collision_penalty, flight_height`
- 第 203 行：根据 env_id 动态计算 state_space 维度（3×N_agents位置 + 3×N_agents速度 + 3×N_landmarks）
- `state()` 方法：为 SimpleSpreadAviary3D 返回真实全局状态（所有位置 + 速度 + landmark 位置）

### 2. YAML 配置文件 — 创建 4 个新文件

路径格式：`xuance/xuance/configs/{algo}/Drones/SimpleSpreadAviary3D.yaml`

算法：mappo, maddpg, masac, matd3

关键配置：
- `env_name: "Drones"` / `env_id: "SimpleSpreadAviary3D"`
- `obs_type: "kin"` / `act_type: "vel"` / `num_drones: 3`
- `num_landmarks: 3` / `world_size: 2.0` / `collision_radius: 0.15` / `collision_penalty: 1.0` / `flight_height: 1.0`
- `max_episode_steps: 1600`（与原 SimpleSpreadDrones3DEnv 一致）
- `render: False` / `sleep: 0.01`
- `activation_action: "tanh"`（VEL 动作在 [-1,1]，不是 MPE 的 sigmoid [0,1]）
- 移除 MPE 物理参数（dt, damping, mass, force_sensitivity）— PyBullet 环境不需要
- 网络结构、学习率等从各算法的 DroneSpread3D 配置复制

### 3. 不需要修改的文件
- `multi_agent_env/__init__.py` — "Drones" 已注册
- `benchmark_flat_3d.py` — 用户通过 `--env Drones --env-id SimpleSpreadAviary3D` 覆盖默认值

## 验证方法

1. 运行：`python benchmark_flat_3d.py --env Drones --env-id SimpleSpreadAviary3D --algo mappo --mode train`
2. 确认环境创建成功（无 import/init 错误）
3. 确认每架无人机的观测维度正确（3 agents, 3 landmarks 时为 84 维）
4. 确认动作维度为 4/无人机（VEL 类型）
5. 确认奖励是共享的且形状正确
6. 确认 1600 步后回合截断
7. 确认无人机运动无空间约束（不因出界截断）
