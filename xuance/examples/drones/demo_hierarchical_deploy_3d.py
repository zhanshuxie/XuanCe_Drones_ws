"""
三维分层部署脚本 — MAPPO 三维高层规划器 + PPO 低层导航控制器

将训练好的 MAPPO 三维规划器（SimpleSpreadDrones3DPlanner 环境训练）与
Navigate PPO 控制器（单无人机 A→B 飞行）拼接，在 PyBullet 中
运行 3 架无人机无碰撞覆盖 3 个目标点。

与二维版 (demo_hierarchical_deploy.py) 的核心区别：
  - 规划器输出三维位移 (dx, dy, dz)，高度不再固定
  - 目标点高度在 [0.5, 4.5] m 中随机生成

坐标映射:
  规划器 XY ∈ [-1, 1]  ↔  物理 XY ∈ [-scale, scale] 米  (scale=2.0)
  规划器 Z  ∈ [-1, 1]  ↔  物理 Z  ∈ [0.5, 4.5] 米       (z = planner_z * 2.0 + 2.5)

用法:
    python demo_hierarchical_deploy_3d.py --planner-model results/mappo/SimpleSpreadDrones3DPlanner/best_model/best_model.pth --navigate-model results/ppo/NavigateAviary/best_model/best_model.pth --render True
"""
import argparse
import numpy as np
import time
from argparse import Namespace

import pybullet as p
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary


# ---------------------------------------------------------------------------
# Coordinate mapping helpers
# ---------------------------------------------------------------------------
def planner_z_to_physical(planner_z):
    """规划器 Z ∈ [-1, 1] → 物理 Z ∈ [0.5, 4.5]"""
    return planner_z * 2.0 + 2.5


def physical_z_to_planner(physical_z):
    """物理 Z ∈ [0.5, 4.5] → 规划器 Z ∈ [-1, 1]"""
    return (physical_z - 2.5) / 2.0


# ---------------------------------------------------------------------------
# Deployment Aviary — observations match NavigateAviary format
# ---------------------------------------------------------------------------
class DeployAviary(BaseRLAviary):
    """
    Thin 3-drone aviary for hierarchical deployment.

    Observations per drone: [rpy(3), vel(3), ang_vel(3), rel_pos_to_waypoint(3)]
    plus action buffer, identical to NavigateAviary.
    """

    def __init__(self, waypoints, **kwargs):
        self.waypoints = np.array(waypoints, dtype=np.float32)
        super().__init__(**kwargs)
        self.SPEED_LIMIT = 0.3  # match NavigateAviary

    def set_waypoints(self, waypoints):
        self.waypoints = np.array(waypoints, dtype=np.float32)

    # ---- abstract implementations (only _computeObs matters) ----

    def _computeObs(self):
        obs = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            obs[i] = np.hstack([
                state[7:10],                       # roll, pitch, yaw
                state[10:13],                      # vx, vy, vz
                state[13:16],                      # wx, wy, wz
                self.waypoints[i] - state[0:3],    # relative pos to waypoint
            ])
        ret = obs.astype("float32")
        for k in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([
                ret,
                np.array([self.action_buffer[k][j, :] for j in range(self.NUM_DRONES)]),
            ])
        return ret

    def _computeReward(self):
        return np.zeros(self.NUM_DRONES)

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_planner_obs(drone_pos_3d, drone_vel_3d, landmark_pos_3d, n):
    """Construct observation dict matching SimpleSpreadDrones3DPlannerEnv format."""
    obs = {}
    for i in range(n):
        own_pos = drone_pos_3d[i]
        own_vel = drone_vel_3d[i]
        rel_lm = (landmark_pos_3d - own_pos).flatten()
        other_idx = [j for j in range(n) if j != i]
        rel_ag = (drone_pos_3d[other_idx] - own_pos).flatten()
        obs[f"agent_{i}"] = np.concatenate(
            [own_pos, own_vel, rel_lm, rel_ag]
        ).astype(np.float32)
    return obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    pa = argparse.ArgumentParser("Hierarchical Deploy 3D: MAPPO 3D Planner + Navigate PPO")
    pa.add_argument("--planner-model", type=str, required=True,
                    help="Path to trained MAPPO 3D planner (.pth or directory)")
    pa.add_argument("--navigate-model", type=str, required=True,
                    help="Path to trained Navigate PPO (.pth or directory)")
    pa.add_argument("--device", type=str, default="cuda:0")
    pa.add_argument("--render", type=lambda x: x.lower() == "true", default=True)
    pa.add_argument("--scale", type=float, default=2.0,
                    help="Planner XY 1.0 = this many physical meters")
    pa.add_argument("--max-high-steps", type=int, default=25)
    pa.add_argument("--max-low-steps", type=int, default=32,
                    help="Low-level steps per high-level planning step")
    pa.add_argument("--seed", type=int, default=10)
    return pa.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    from xuance import get_runner

    # ---- 1. Load MAPPO 3D planner agent ----
    print("[1/4] Loading MAPPO 3D planner …")
    planner_runner = get_runner(
        algo="mappo", env="DroneSpreadPlanner3D", env_id="SimpleSpreadDrones3DPlanner",
        parser_args=Namespace(parallels=1, device=args.device),
    )
    planner_agent = planner_runner.agent
    planner_agent.load_model(args.planner_model)
    planner_runner.envs.close()

    # ---- 2. Load Navigate PPO agent ----
    print("[2/4] Loading Navigate PPO controller …")
    nav_runner = get_runner(
        algo="ppo", env="Drone", env_id="NavigateAviary",
        parser_args=Namespace(parallels=1, device=args.device, render=False),
    )
    nav_agent = nav_runner.agent
    nav_agent.load_model(args.navigate_model)
    nav_runner.envs.close()

    # ---- 3. Initialise positions ----
    print("[3/4] Setting up world …")
    N = 3                       # drones and landmarks
    ws = 1.0                    # planner world size
    max_step = 0.15             # planner max displacement per step
    scale = args.scale

    # 无人机初始位置：XYZ 均在规划器坐标系 [-1, 1] 中随机
    drone_pos_3d = rng.uniform(-ws, ws, (N, 3)).astype(np.float32)
    drone_vel_3d = np.zeros((N, 3), dtype=np.float32)

    # 物理空间初始位置：XY = planner * scale, Z = 线性映射
    init_xyz = np.column_stack([
        drone_pos_3d[:, 0:2] * scale,
        planner_z_to_physical(drone_pos_3d[:, 2]),
    ])

    # 目标点：XY 在物理空间 [-2, 2] 随机，Z 在物理空间 [0.5, 4.5] 随机
    lm_xy_phys = rng.uniform(-scale, scale, (N, 2)).astype(np.float32)
    lm_z_phys = rng.uniform(0.5, 4.5, (N,)).astype(np.float32)
    lm_xyz = np.column_stack([lm_xy_phys, lm_z_phys])

    # 目标点转换到规划器坐标系
    lm_pos_3d = np.column_stack([
        lm_xy_phys / scale,
        physical_z_to_planner(lm_z_phys),
    ]).astype(np.float32)

    print(f"  Drones  (m) : {np.round(init_xyz, 3).tolist()}")
    print(f"  Targets (m) : {np.round(lm_xyz, 3).tolist()}")

    # ---- 4. Create deployment aviary ----
    print("[4/4] Launching PyBullet …")
    aviary = DeployAviary(
        waypoints=init_xyz.copy(),
        drone_model=DroneModel.CF2X,
        num_drones=N,
        initial_xyzs=init_xyz,
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=30,
        gui=args.render,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
    )
    aviary.reset()

    # draw landmark spheres
    if args.render:
        colours = [[1, 0, 0, 0.8], [0, 1, 0, 0.8], [0, 0, 1, 0.8]]
        for i, lm in enumerate(lm_xyz):
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.05,
                rgbaColor=colours[i % len(colours)],
                physicsClientId=aviary.CLIENT,
            )
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                              basePosition=lm.tolist(),
                              physicsClientId=aviary.CLIENT)

    # ---- 5. Hierarchical control loop ----
    # 规划器维护的虚拟 3D agent 位置（初始 = 无人机初始 3D 位置）
    planner_pos_3d = drone_pos_3d.copy()   # (N, 3), planner frame [-1, 1]

    cover_th = 0.2      # landmark 覆盖判定阈值 (m)

    print("\n--- Starting hierarchical control ---")
    for hi in range(args.max_high_steps):
        # 5-a  读取无人机实际 3D 位置，转换到规划器坐标系
        for i in range(N):
            phys_pos = aviary._getDroneStateVector(i)[0:3]
            drone_pos_3d[i, 0:2] = phys_pos[0:2] / scale
            drone_pos_3d[i, 2] = physical_z_to_planner(phys_pos[2])

        # 5-b  构建规划器观测（用无人机实际位置）
        planner_obs = build_planner_obs(drone_pos_3d, drone_vel_3d, lm_pos_3d, N)

        # 5-c  MAPPO 推理
        out = planner_agent.action([planner_obs], test_mode=True)
        act_dict = out["actions"][0]       # {agent_i: np.ndarray}

        # 5-d  3D 位移更新虚拟位置 → 映射为 3D 绝对航点
        wp3 = np.zeros((N, 3))
        for i in range(N):
            d3 = np.clip(act_dict[f"agent_{i}"][:3], -1, 1) * max_step
            drone_vel_3d[i] = d3           # 记录速度供下次观测

            # 更新规划器内部虚拟位置
            planner_pos_3d[i] = np.clip(
                drone_pos_3d[i] + d3, -ws, ws
            )

            # 航点 = 虚拟位置映射到物理空间
            wp3[i, 0:2] = np.clip(planner_pos_3d[i, 0:2] * scale, -scale, scale)
            wp3[i, 2] = np.clip(planner_z_to_physical(planner_pos_3d[i, 2]), 0.5, 4.5)

        aviary.set_waypoints(wp3)

        # 5-e  低层控制器：固定运行 max_low_steps 步，到达航点的无人机悬停
        reach_th = 0.05  # 到达判定阈值 (m)，距航点 < 0.05m 则发零速度悬停
        for _ in range(args.max_low_steps):
            obs = aviary._computeObs()              # (N, obs_dim)
            actions = np.zeros((N, 4), dtype=np.float32)

            # 未到达航点的无人机走 Navigate 推理
            pending = [i for i in range(N)
                       if np.linalg.norm(aviary._getDroneStateVector(i)[0:3] - wp3[i]) >= reach_th]
            if pending:
                obs_pending = obs[pending]           # (n_pending, obs_dim)
                obs_n = nav_agent._process_observation(obs_pending)
                nav_out = nav_agent.action(obs_n)
                for k, i in enumerate(pending):
                    actions[i] = nav_out["actions"][k]
            # 已到达的 actions[i] 保持 [0,0,0,0] → 悬停

            aviary.step(actions)
            if args.render:
                time.sleep(1.0 / 240)
        low_used = args.max_low_steps

        # 5-f  状态打印
        min_dists = []
        for lm in lm_xyz:
            min_dists.append(
                min(np.linalg.norm(aviary._getDroneStateVector(j)[0:3] - lm)
                    for j in range(N))
            )
        drone_heights = [f"{aviary._getDroneStateVector(i)[2]:.2f}" for i in range(N)]
        print(f"  high {hi+1:2d}/{args.max_high_steps} | "
              f"low {low_used:3d} | "
              f"heights {drone_heights} | "
              f"lm dists {[f'{d:.3f}' for d in min_dists]}")

        if all(d < cover_th for d in min_dists):
            print(f"\n=== All landmarks covered at step {hi+1}! ===")
            break

    # ---- 6. Report ----
    print("\nFinal positions (m):")
    for i in range(N):
        pos = aviary._getDroneStateVector(i)[0:3]
        print(f"  Drone    {i}: {np.round(pos, 3).tolist()}")
    for i, lm in enumerate(lm_xyz):
        print(f"  Landmark {i}: {np.round(lm, 3).tolist()}")

    aviary.close()
    print("Done.")


if __name__ == "__main__":
    main()
