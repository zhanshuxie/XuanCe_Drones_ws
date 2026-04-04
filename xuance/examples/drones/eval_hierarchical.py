"""
分层部署评估脚本 — 消融实验用

加载训练好的MAPPO规划器 + PPO导航器的checkpoint，
在PyBullet中运行分层部署，计算MPE simple_spread奖励。

对每个checkpoint（100k, 200k, ..., 10000k）运行5轮，
计算平均奖励，输出CSV。

消融变量: --max-low-steps (8/16/32/64)

用法:
    python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 8  --output results_mls8.csv
    python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 16 --output results_mls16.csv
    python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 32 --output results_mls32.csv
    python eval_hierarchical.py --ckpt-root checkpoints --max-low-steps 64 --output results_mls64.csv
"""
import os
import csv
import argparse
import numpy as np
from argparse import Namespace

import pybullet as p
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary


# ---------------------------------------------------------------------------
# DeployAviary — 与 demo_hierarchical_deploy.py 完全相同
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
# Helpers — 与 demo_hierarchical_deploy.py 完全相同
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


def compute_reward_3d(aviary, lm_xyz, n, collision_radius=0.15, collision_penalty=1.0):
    """
    计算MPE simple_spread奖励（3D物理坐标二范数）。

    reward = -Σ_landmark min_agent_dist(landmark) - collision_penalty * n_collisions
    """
    drone_pos_3d = np.array([
        aviary._getDroneStateVector(i)[0:3] for i in range(n)
    ])

    reward = 0.0
    # 1) 覆盖奖励：每个landmark到最近无人机的3D距离
    for lm in lm_xyz:
        dists = np.linalg.norm(drone_pos_3d - lm, axis=1)
        reward -= float(np.min(dists))
    # 2) 碰撞惩罚：无人机之间的3D距离
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(drone_pos_3d[i] - drone_pos_3d[j]) < collision_radius:
                reward -= collision_penalty
    return reward


# ---------------------------------------------------------------------------
# 单轮评估
# ---------------------------------------------------------------------------
def run_one_episode(planner_agent, nav_agent, rng,
                    max_low_steps, max_high_steps=25,
                    scale=2.0, flight_height=1.0):
    """
    运行一轮分层部署，返回该episode的累积奖励。

    内部逻辑与 demo_hierarchical_deploy.py 完全一致，
    额外在每个高层步骤结束后计算并累加奖励。
    """
    N = 3
    ws = 1.0
    max_step = 0.15

    # 随机初始化位置
    drone_pos_2d = rng.uniform(-ws, ws, (N, 2)).astype(np.float32)
    lm_pos_2d = rng.uniform(-ws, ws, (N, 2)).astype(np.float32)
    drone_vel_2d = np.zeros((N, 2), dtype=np.float32)

    init_xyz = np.column_stack([drone_pos_2d * scale, np.full(N, flight_height)])
    lm_xyz = np.column_stack([lm_pos_2d * scale, np.full(N, flight_height)])

    # 创建部署环境
    aviary = DeployAviary(
        waypoints=init_xyz.copy(),
        drone_model=DroneModel.CF2X,
        num_drones=N,
        initial_xyzs=init_xyz,
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=30,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
    )
    aviary.reset()

    # 分层控制循环
    planner_pos_2d = drone_pos_2d.copy()
    ep_reward = 0.0

    for hi in range(max_high_steps):
        # 5-a 读取无人机实际2D位置
        for i in range(N):
            drone_pos_2d[i] = aviary._getDroneStateVector(i)[0:2] / scale

        # 5-b 构建规划器观测
        planner_obs = build_planner_obs(drone_pos_2d, drone_vel_2d, lm_pos_2d, N)

        # 5-c MAPPO推理
        out = planner_agent.action([planner_obs], test_mode=True)
        act_dict = out["actions"][0]

        # 5-d 更新虚拟位置 → 映射为3D航点
        wp3 = np.zeros((N, 3))
        for i in range(N):
            d2 = np.clip(act_dict[f"agent_{i}"][:2], -1, 1) * max_step
            drone_vel_2d[i] = d2

            planner_pos_2d[i] = np.clip(
                drone_pos_2d[i] + d2, -ws, ws
            )

            wp3[i] = np.array([
                planner_pos_2d[i, 0] * scale,
                planner_pos_2d[i, 1] * scale,
                flight_height,
            ])

        aviary.set_waypoints(wp3)

        # 5-e 低层控制器
        reach_th = 0.05
        for _ in range(max_low_steps):
            obs = aviary._computeObs()
            actions = np.zeros((N, 4), dtype=np.float32)

            pending = [i for i in range(N)
                       if np.linalg.norm(
                           aviary._getDroneStateVector(i)[0:3] - wp3[i]
                       ) >= reach_th]
            if pending:
                obs_pending = obs[pending]
                obs_n = nav_agent._process_observation(obs_pending)
                nav_out = nav_agent.action(obs_n)
                for k, i in enumerate(pending):
                    actions[i] = nav_out["actions"][k]

            aviary.step(actions)

        # 5-f 状态评估
        # min_dists = []
        # for lm in lm_xyz:
        #     min_dists.append(
        #         min(np.linalg.norm(aviary._getDroneStateVector(j)[0:3] - lm)
        #             for j in range(N))
        #     )
        # # 提前终止逻辑先注释掉
        # if all(d < 0.2 for d in min_dists):
        #     break

        # 计算该高层步骤的奖励（3D物理坐标二范数）
        step_reward = compute_reward_3d(aviary, lm_xyz, N)
        ep_reward += step_reward

    aviary.close()
    return ep_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    pa = argparse.ArgumentParser("Hierarchical Deployment Evaluation for Ablation")
    pa.add_argument("--ckpt-root", type=str, default="checkpoints",
                    help="Root directory containing planner/ and navigator/ checkpoints")
    pa.add_argument("--max-low-steps", type=int, default=32,
                    help="Low-level steps per high-level step (ablation variable: 8/16/32/64)")
    pa.add_argument("--max-high-steps", type=int, default=25,
                    help="Max high-level planning steps per episode")
    pa.add_argument("--scale", type=float, default=2.0)
    pa.add_argument("--flight-height", type=float, default=1.0)
    pa.add_argument("--episodes", type=int, default=5,
                    help="Number of episodes per checkpoint")
    pa.add_argument("--seed", type=int, default=42,
                    help="Base random seed for reproducibility")
    pa.add_argument("--device", type=str, default="cuda:0")
    pa.add_argument("--output", type=str, default=None,
                    help="Output CSV path (default: results_mls{max-low-steps}.csv)")
    return pa.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = f"results_mls{args.max_low_steps}.csv"

    print("=" * 60)
    print(f"Hierarchical Evaluation | max-low-steps = {args.max_low_steps}")
    print("=" * 60)

    from xuance import get_runner

    # ---- 1. 创建agent（仅一次） ----
    print("[1/3] Loading MAPPO planner agent ...")
    planner_runner = get_runner(
        algo="mappo", env="DroneSpread", env_id="SimpleSpreadDrones",
        parser_args=Namespace(parallels=1, device=args.device),
    )
    planner_agent = planner_runner.agent
    planner_runner.envs.close()

    print("[2/3] Loading PPO navigator agent ...")
    nav_runner = get_runner(
        algo="ppo", env="Drone", env_id="NavigateAviary",
        parser_args=Namespace(parallels=1, device=args.device, render=False),
    )
    nav_agent = nav_runner.agent
    nav_runner.envs.close()

    # ---- 2. 准备CSV ----
    print(f"[3/3] Starting evaluation → {args.output}")
    print(f"  Checkpoints: {os.path.abspath(args.ckpt_root)}")
    print(f"  Episodes per checkpoint: {args.episodes}")
    print(f"  Base seed: {args.seed}")
    print()

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "avg_reward", "std_reward"])

    # ---- 3. 遍历100个checkpoint ----
    for ckpt_idx in range(100):
        step_k = (ckpt_idx + 1) * 100  # 100, 200, ..., 10000

        planner_path = os.path.join(
            args.ckpt_root, "planner", f"{step_k}k", "model.pth"
        )
        nav_path = os.path.join(
            args.ckpt_root, "navigator", f"{step_k}k", "model.pth"
        )

        # 检查文件是否存在
        if not os.path.isfile(planner_path):
            print(f"  [SKIP] {planner_path} not found")
            continue
        if not os.path.isfile(nav_path):
            print(f"  [SKIP] {nav_path} not found")
            continue

        # 加载模型
        planner_agent.load_model(planner_path)
        nav_agent.load_model(nav_path)  # 自动恢复obs_rms

        # 运行多轮评估
        episode_rewards = []
        for ep in range(args.episodes):
            # 确定性种子：同一 ckpt_idx + ep 组合始终相同
            ep_seed = args.seed + ckpt_idx * args.episodes + ep
            rng = np.random.default_rng(ep_seed)

            ep_reward = run_one_episode(
                planner_agent, nav_agent, rng,
                max_low_steps=args.max_low_steps,
                max_high_steps=args.max_high_steps,
                scale=args.scale,
                flight_height=args.flight_height,
            )
            episode_rewards.append(ep_reward)

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # 写入CSV
        with open(args.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step_k, f"{avg_reward:.4f}", f"{std_reward:.4f}"])

        print(f"  ckpt {step_k:>5d}k | "
              f"avg_reward={avg_reward:>8.3f} ± {std_reward:.3f} | "
              f"episodes={episode_rewards}")

    print(f"\nEvaluation complete. Results saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
