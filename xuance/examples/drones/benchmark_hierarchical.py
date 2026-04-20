"""
分层算法 Benchmark 脚本 — 与 flat benchmark 对齐的奖励计算

与 eval_hierarchical.py 的唯一区别：
  奖励在每个低层 step 计算一次（共 25×max_low_steps 次/episode），
  而非每个高层决策步计算一次（25次/episode）。
  这样 episode return 与 benchmark_flat_3d.py 可直接比较。

用法:
    python benchmark_hierarchical.py --ckpt-root checkpoints --max-low-steps 64 --output benchmark_hier.csv
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
# DeployAviary — 与 eval_hierarchical.py 完全相同
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
# 确定性推理：评估时使用分布均值，避免随机采样带来的方差
# ---------------------------------------------------------------------------
def planner_deterministic_action(planner_agent, obs_dict):
    """MAPPO 确定性推理：取高斯分布均值而非随机采样。"""
    obs_input, agents_id, avail_actions_input = planner_agent._build_inputs([obs_dict], None)
    _, pi_dists = planner_agent.policy(observation=obs_input,
                                       agent_ids=agents_id,
                                       avail_actions=avail_actions_input,
                                       rnn_hidden=None)
    key = planner_agent.agent_keys[0]
    actions_det = pi_dists[key].deterministic_sample()
    n_agents = planner_agent.n_agents
    actions_out = actions_det.reshape(1, n_agents, -1)
    act_dict = {k: actions_out[0, i].cpu().detach().numpy()
                for i, k in enumerate(planner_agent.agent_keys)}
    return act_dict


def navigator_deterministic_action(nav_agent, obs):
    """PPO 确定性推理：取高斯分布均值而非随机采样。"""
    _, policy_dists, _ = nav_agent.policy(obs)
    actions = policy_dists.deterministic_sample()
    return actions.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Helpers — 与 eval_hierarchical.py 完全相同
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
# 批量评估（多个并行环境）
# 与 eval_hierarchical.py 的唯一区别：奖励在每个低层 step 计算
# ---------------------------------------------------------------------------
def run_episodes_batch(planner_agent, nav_agent, scenarios_batch,
                       max_low_steps, max_high_steps=25,
                       scale=2.0, flight_height=1.0):
    """
    同时运行多个 episode（多个 PyBullet DIRECT 实例并存）。
    奖励在每个低层 step 计算一次，与 flat benchmark 对齐。

    Parameters
    ----------
    scenarios_batch : list of (drone_pos_2d, lm_pos_2d)
        每个元素是一个场景的预生成位置

    Returns
    -------
    list of (ep_reward, avg_arrival_rate, final_coverage_dist)
    """
    N = 3
    ws = 1.0
    max_step = 0.15
    reach_th = 0.05
    n_envs = len(scenarios_batch)

    # ---- 初始化所有环境 ----
    envs = []
    for drone_pos_2d_init, lm_pos_2d_init in scenarios_batch:
        drone_pos_2d = drone_pos_2d_init.copy()
        lm_pos_2d = lm_pos_2d_init.copy()
        init_xyz = np.column_stack([drone_pos_2d * scale, np.full(N, flight_height)])
        lm_xyz = np.column_stack([lm_pos_2d * scale, np.full(N, flight_height)])

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
        envs.append({
            'aviary': aviary,
            'drone_pos_2d': drone_pos_2d,
            'lm_pos_2d': lm_pos_2d,
            'drone_vel_2d': np.zeros((N, 2), dtype=np.float32),
            'planner_pos_2d': drone_pos_2d.copy(),
            'lm_xyz': lm_xyz,
            'wp3': np.zeros((N, 3)),
            'ep_reward': 0.0,
            'arrival_rates': [],
        })

    # ---- 分层控制循环（所有环境同步推进） ----
    for hi in range(max_high_steps):
        # 高层：逐环境做规划器推理 + 设置航点
        for e in envs:
            av = e['aviary']
            dp, dv, pp = e['drone_pos_2d'], e['drone_vel_2d'], e['planner_pos_2d']

            for i in range(N):
                dp[i] = av._getDroneStateVector(i)[0:2] / scale

            planner_obs = build_planner_obs(dp, dv, e['lm_pos_2d'], N)
            act_dict = planner_deterministic_action(planner_agent, planner_obs)

            wp3 = e['wp3']
            for i in range(N):
                d2 = np.clip(act_dict[f"agent_{i}"][:2], -1, 1) * max_step
                dv[i] = d2
                pp[i] = np.clip(dp[i] + d2, -ws, ws)
                wp3[i] = [pp[i, 0] * scale, pp[i, 1] * scale, flight_height]
            av.set_waypoints(wp3)

        # 低层：所有环境同步步进，navigator 批量推理
        for _ in range(max_low_steps):
            # 收集所有环境中未到达的无人机观测
            all_obs = []
            idx_map = []  # (env_idx, drone_idx)
            for ei, e in enumerate(envs):
                av = e['aviary']
                obs = av._computeObs()
                e['_obs'] = obs
                e['_actions'] = np.zeros((N, 4), dtype=np.float32)
                for i in range(N):
                    if np.linalg.norm(av._getDroneStateVector(i)[0:3] - e['wp3'][i]) >= reach_th:
                        all_obs.append(obs[i])
                        idx_map.append((ei, i))

            # 批量 navigator 推理
            if all_obs:
                obs_batch = np.array(all_obs)
                obs_n = nav_agent._process_observation(obs_batch)
                nav_actions = navigator_deterministic_action(nav_agent, obs_n)
                for k, (ei, di) in enumerate(idx_map):
                    envs[ei]['_actions'][di] = nav_actions[k]

            # 所有环境执行动作
            for e in envs:
                e['aviary'].step(e['_actions'])

            # ★ 每个低层 step 计算奖励（与 flat benchmark 对齐）
            for e in envs:
                e['ep_reward'] += compute_reward_3d(e['aviary'], e['lm_xyz'], N)

        # 统计到达率（保留在高层循环）
        for e in envs:
            av = e['aviary']
            arrived = sum(1 for i in range(N)
                          if np.linalg.norm(av._getDroneStateVector(i)[0:3] - e['wp3'][i]) < reach_th)
            e['arrival_rates'].append(arrived / N)

    # ---- 收集结果并关闭环境 ----
    results = []
    for e in envs:
        av = e['aviary']
        drone_pos_final = np.array([av._getDroneStateVector(i)[0:3] for i in range(N)])
        final_dists = [float(np.min(np.linalg.norm(drone_pos_final - lm, axis=1)))
                       for lm in e['lm_xyz']]
        results.append((e['ep_reward'], np.mean(e['arrival_rates']), np.mean(final_dists)))
        av.close()

    return results


# ---------------------------------------------------------------------------
# Main — 与 eval_hierarchical.py 完全相同
# ---------------------------------------------------------------------------
def parse_args():
    pa = argparse.ArgumentParser("Hierarchical Benchmark (per-step reward)")
    pa.add_argument("--ckpt-root", type=str, default="checkpoints",
                    help="Root directory containing planner/ and navigator/ checkpoints")
    pa.add_argument("--max-low-steps", type=int, default=64,
                    help="Low-level steps per high-level step")
    pa.add_argument("--max-high-steps", type=int, default=25,
                    help="Max high-level planning steps per episode")
    pa.add_argument("--scale", type=float, default=2.0)
    pa.add_argument("--flight-height", type=float, default=1.0)
    pa.add_argument("--episodes", type=int, default=10,
                    help="Number of episodes per checkpoint")
    pa.add_argument("--parallels", type=int, default=5,
                    help="Number of parallel environments")
    pa.add_argument("--seed", type=int, default=1,
                    help="Base random seed (matches flat benchmark env_seed)")
    pa.add_argument("--device", type=str, default="cuda:0")
    pa.add_argument("--nav-ckpt", type=str, default="checkpoints/navigator/best/model.pth",
                    help="Fixed navigator checkpoint path (best model)")
    pa.add_argument("--output", type=str, default=None,
                    help="Output CSV path (default: benchmark_hier_mls{max-low-steps}.csv)")
    return pa.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = f"benchmark_hier_mls{args.max_low_steps}.csv"

    print("=" * 60)
    print(f"Hierarchical Benchmark | max-low-steps = {args.max_low_steps}")
    print(f"  Reward computed per low-level step (aligned with flat benchmark)")
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

    # 如果指定了固定 navigator checkpoint，一次性加载
    if args.nav_ckpt:
        print(f"  Navigator fixed at: {args.nav_ckpt}")
        nav_agent.load_model(args.nav_ckpt)

    # ---- 2. 预生成所有场景（复现 flat benchmark 测试环境的 rng 行为） ----
    # flat benchmark: 5 个并行测试 env，seed 分别为 env_seed+0 ~ env_seed+4
    # 每次 eval: 每个 env reset 3 次 (2 轮计分 + 1 次 extra auto-reset)
    # rng 状态在 eval 间累积，不重置
    N = 3
    ws = 2.0                       # world_size
    n_test_parallel = 5            # 与 flat benchmark 的 test_parallels 对齐
    episodes_per_env = args.episodes // n_test_parallel  # = 2
    resets_per_eval = episodes_per_env + 1  # 2 轮计分 + 1 次 extra = 3

    # 创建 5 个独立的 rng，与 flat 的 5 个测试 env 一一对应
    rngs = [np.random.default_rng(args.seed + i) for i in range(n_test_parallel)]

    # 跳过 flat benchmark step=0 的初始 eval（hierarchical 没有这次 eval）
    for rng in rngs:
        for _ in range(resets_per_eval):
            rng.uniform(-ws, ws, (N, 2))  # drone_xy
            rng.uniform(-ws, ws, (N, 2))  # lm_xy

    # 为 100 个 checkpoint 生成场景
    scenarios = {}
    for ckpt_idx in range(100):
        # episode 顺序与 flat 一致: env0_b0, env1_b0, ..., env4_b0,
        #                            env0_b1, env1_b1, ..., env4_b1
        for batch in range(episodes_per_env):
            for env_i in range(n_test_parallel):
                drone_xy = rngs[env_i].uniform(-ws, ws, (N, 2)).astype(np.float32)
                lm_xy = rngs[env_i].uniform(-ws, ws, (N, 2)).astype(np.float32)
                ep_idx = batch * n_test_parallel + env_i
                scenarios[(ckpt_idx, ep_idx)] = (drone_xy / args.scale, lm_xy / args.scale)

        # extra auto-reset（不计分，但推进 rng 状态以保持与 flat 同步）
        for env_i in range(n_test_parallel):
            rngs[env_i].uniform(-ws, ws, (N, 2))
            rngs[env_i].uniform(-ws, ws, (N, 2))

    # 打印前3个场景用于验证一致性（显示物理坐标，方便与 flat benchmark 对比）
    print("[Scenario verification] First 3 scenarios (ckpt=0, physical coords):")
    for ep in range(min(3, args.episodes)):
        d, l = scenarios[(0, ep)]
        print(f"  ep={ep}: drone={( d * args.scale).tolist()}, landmark={(l * args.scale).tolist()}")
    print()

    # ---- 3. 准备CSV ----
    print(f"[3/3] Starting evaluation → {args.output}")
    print(f"  Checkpoints: {os.path.abspath(args.ckpt_root)}")
    print(f"  Episodes per checkpoint: {args.episodes}")
    print(f"  Base seed: {args.seed}")
    print()

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step", "avg_reward", "std_reward",
                  "avg_arrival_rate", "std_arrival_rate",
                  "avg_final_dist", "std_final_dist"]
        header += [f"ep_{i}_reward" for i in range(args.episodes)]
        writer.writerow(header)

    # ---- 4. 遍历100个checkpoint ----
    for ckpt_idx in range(100):
        step_k = (ckpt_idx + 1) * 100  # 100, 200, ..., 10000

        planner_path = os.path.join(
            args.ckpt_root, "planner", f"{step_k}k", "model.pth"
        )

        # 检查 planner checkpoint 是否存在
        if not os.path.isfile(planner_path):
            print(f"  [SKIP] {planner_path} not found")
            continue

        # 加载 planner；navigator 若未固定则按步数加载
        planner_agent.load_model(planner_path)
        if not args.nav_ckpt:
            nav_path = os.path.join(
                args.ckpt_root, "navigator", f"{step_k}k", "model.pth"
            )
            if not os.path.isfile(nav_path):
                print(f"  [SKIP] {nav_path} not found")
                continue
            nav_agent.load_model(nav_path)

        # 运行多轮评估（按批次并行）
        episode_rewards = []
        episode_arrival_rates = []
        episode_final_dists = []
        for batch_start in range(0, args.episodes, args.parallels):
            batch_end = min(batch_start + args.parallels, args.episodes)
            batch_scenarios = [
                (scenarios[(ckpt_idx, ep)][0].copy(), scenarios[(ckpt_idx, ep)][1].copy())
                for ep in range(batch_start, batch_end)
            ]
            batch_results = run_episodes_batch(
                planner_agent, nav_agent, batch_scenarios,
                max_low_steps=args.max_low_steps,
                max_high_steps=args.max_high_steps,
                scale=args.scale,
                flight_height=args.flight_height,
            )
            for ep_reward, ep_arrival, ep_final_dist in batch_results:
                episode_rewards.append(ep_reward)
                episode_arrival_rates.append(ep_arrival)
                episode_final_dists.append(ep_final_dist)

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_arrival = np.mean(episode_arrival_rates)
        std_arrival = np.std(episode_arrival_rates)
        avg_final_dist = np.mean(episode_final_dists)
        std_final_dist = np.std(episode_final_dists)

        # 写入CSV
        with open(args.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [step_k,
                   f"{avg_reward:.4f}", f"{std_reward:.4f}",
                   f"{avg_arrival:.4f}", f"{std_arrival:.4f}",
                   f"{avg_final_dist:.4f}", f"{std_final_dist:.4f}"]
            row += [f"{r:.4f}" for r in episode_rewards]
            writer.writerow(row)

        print(f"  ckpt {step_k:>5d}k | "
              f"reward={avg_reward:>8.3f}±{std_reward:.3f} | "
              f"arrival={avg_arrival:.2f} | "
              f"final_dist={avg_final_dist:.4f}")

    print(f"\nBenchmark complete. Results saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
