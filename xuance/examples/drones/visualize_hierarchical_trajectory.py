"""
分层部署 + 轨迹可视化脚本

运行 MAPPO 高层规划器 + PPO 低层导航控制器，在 PyBullet 中
让 3 架无人机覆盖 3 个目标点，并将飞行轨迹以 3 种颜色绘制成 3D 图。

用法:
    python visualize_hierarchical_trajectory.py
    python visualize_hierarchical_trajectory.py --planner-model <path> --navigate-model <path> --render True
"""
import argparse
import numpy as np
import time
from argparse import Namespace

import pybullet as p
import matplotlib.pyplot as plt
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary


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

    def _computeObs(self):
        obs = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            obs[i] = np.hstack([
                state[7:10],
                state[10:13],
                state[13:16],
                self.waypoints[i] - state[0:3],
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
def build_planner_obs(drone_pos_2d, drone_vel_2d, landmark_pos_2d, n):
    """Construct observation dict matching SimpleSpreadDronesEnv format."""
    obs = {}
    for i in range(n):
        own_pos = drone_pos_2d[i]
        own_vel = drone_vel_2d[i]
        rel_lm = (landmark_pos_2d - own_pos).flatten()
        other_idx = [j for j in range(n) if j != i]
        rel_ag = (drone_pos_2d[other_idx] - own_pos).flatten()
        obs[f"agent_{i}"] = np.concatenate(
            [own_pos, own_vel, rel_lm, rel_ag]
        ).astype(np.float32)
    return obs


def plot_trajectories(trajectories, lm_xyz, save_path, max_low_steps):
    """绘制 3 架无人机的 3D 轨迹图。"""
    colors = ["#e74c3c", "#2ecc71", "#3498db"]  # 红、绿、蓝
    labels = ["Drone 0", "Drone 1", "Drone 2"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, traj in enumerate(trajectories):
        traj = np.array(traj)  # (T, 3)
        # 轨迹线
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=colors[i], linewidth=1.5, label=labels[i])
        # 起始位置 — 圆形
        ax.scatter(*traj[0], color=colors[i], marker="o", s=80,
                   edgecolors="black", zorder=5)
        # 终止位置 — 星形
        ax.scatter(*traj[-1], color=colors[i], marker="*", s=150,
                   edgecolors="black", zorder=5)

    # 目标地标 — 菱形 (颜色独立于无人机)
    target_map = [
        (1, "#e74c3c", "Target 0"),  # 原绿色菱形 → 红色
        (2, "#2ecc71", "Target 1"),  # 原蓝色菱形 → 绿色
        (0, "#3498db", "Target 2"),  # 原红色菱形 → 蓝色
    ]
    for lm_i, tc, tl in target_map:
        ax.scatter(*lm_xyz[lm_i], color=tc, marker="D", s=120,
                   edgecolors="black", linewidths=1.2, zorder=5,
                   label=tl)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Hierarchical RL — UAV Trajectories MLS={max_low_steps}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\n轨迹图已保存至 {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    pa = argparse.ArgumentParser("Hierarchical Deploy + Trajectory Visualization")
    pa.add_argument("--planner-model", type=str,
                    default="results/mappo/SimpleSpreadDrones/best_model/best_model.pth",
                    help="Path to trained MAPPO planner (.pth or directory)")
    pa.add_argument("--navigate-model", type=str,
                    default="results/ppo/NavigateAviary/best_model/best_model.pth",
                    help="Path to trained Navigate PPO (.pth or directory)")
    pa.add_argument("--device", type=str, default="cuda:0")
    pa.add_argument("--render", type=lambda x: x.lower() == "true", default=True)
    pa.add_argument("--scale", type=float, default=2.0,
                    help="Planner 1.0 = this many physical meters")
    pa.add_argument("--flight-height", type=float, default=1.0)
    pa.add_argument("--max-high-steps", type=int, default=25)
    pa.add_argument("--max-low-steps", type=int, default=16,
                    help="Low-level steps per high-level planning step")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--save-path", type=str, default="trajectory_plot.png",
                    help="Output path for the trajectory plot image")
    return pa.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    from xuance import get_runner

    # ---- 1. Load MAPPO planner agent ----
    print("[1/4] Loading MAPPO planner …")
    planner_runner = get_runner(
        algo="mappo", env="DroneSpread", env_id="SimpleSpreadDrones",
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
    N = 3
    ws = 1.0
    max_step = 0.15
    scale = args.scale
    height = args.flight_height

    drone_pos_2d = rng.uniform(-ws, ws, (N, 2)).astype(np.float32)
    lm_pos_2d = rng.uniform(-ws, ws, (N, 2)).astype(np.float32)
    drone_vel_2d = np.zeros((N, 2), dtype=np.float32)

    init_xyz = np.column_stack([drone_pos_2d * scale, np.full(N, height)])
    lm_xyz = np.column_stack([lm_pos_2d * scale, np.full(N, height)])

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

    # ---- 轨迹记录初始化 ----
    trajectories = [[] for _ in range(N)]
    for i in range(N):
        trajectories[i].append(aviary._getDroneStateVector(i)[0:3].copy())

    # ---- 5. Hierarchical control loop ----
    planner_pos_2d = drone_pos_2d.copy()
    cover_th = 0.2

    print("\n--- Starting hierarchical control ---")
    for hi in range(args.max_high_steps):
        # 5-a  读取无人机实际 2D 位置
        for i in range(N):
            drone_pos_2d[i] = aviary._getDroneStateVector(i)[0:2] / scale

        # 5-b  构建规划器观测
        planner_obs = build_planner_obs(drone_pos_2d, drone_vel_2d, lm_pos_2d, N)

        # 5-c  MAPPO 推理
        out = planner_agent.action([planner_obs], test_mode=True)
        act_dict = out["actions"][0]

        # 5-d  2D 位移更新虚拟位置 → 映射为 3D 绝对航点
        wp3 = np.zeros((N, 3))
        for i in range(N):
            d2 = np.clip(act_dict[f"agent_{i}"][:2], -1, 1) * max_step
            drone_vel_2d[i] = d2
            planner_pos_2d[i] = np.clip(drone_pos_2d[i] + d2, -ws, ws)
            wp3[i] = np.array([
                planner_pos_2d[i, 0] * scale,
                planner_pos_2d[i, 1] * scale,
                height,
            ])

        aviary.set_waypoints(wp3)

        # 5-e  低层控制器
        reach_th = 0.05
        for _ in range(args.max_low_steps):
            obs = aviary._computeObs()
            actions = np.zeros((N, 4), dtype=np.float32)

            pending = [i for i in range(N)
                       if np.linalg.norm(aviary._getDroneStateVector(i)[0:3] - wp3[i]) >= reach_th]
            if pending:
                obs_pending = obs[pending]
                obs_n = nav_agent._process_observation(obs_pending)
                nav_out = nav_agent.action(obs_n)
                for k, i in enumerate(pending):
                    actions[i] = nav_out["actions"][k]

            aviary.step(actions)

            # ---- 记录轨迹 ----
            for i in range(N):
                trajectories[i].append(aviary._getDroneStateVector(i)[0:3].copy())

            if args.render:
                time.sleep(1.0 / 240)

        # 5-f  状态打印
        min_dists = []
        for lm in lm_xyz:
            min_dists.append(
                min(np.linalg.norm(aviary._getDroneStateVector(j)[0:3] - lm)
                    for j in range(N))
            )
        drone_heights = [f"{aviary._getDroneStateVector(i)[2]:.2f}" for i in range(N)]
        print(f"  high {hi+1:2d}/{args.max_high_steps} | "
              f"low {args.max_low_steps:3d} | "
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

    # ---- 7. 绘制轨迹图 ----
    plot_trajectories(trajectories, lm_xyz, args.save_path, args.max_low_steps)

    print("Done.")


if __name__ == "__main__":
    main()
