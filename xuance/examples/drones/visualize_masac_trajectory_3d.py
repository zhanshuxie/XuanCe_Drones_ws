"""
MaSAC 最佳模型 3D 轨迹可视化脚本

加载 MaSAC 算法在 SimpleSpreadAviary3D 环境中训练的最佳模型，
运行一个 episode，记录 3 架无人机的 3D 飞行轨迹并绘图。

用法:
    python visualize_masac_trajectory_3d.py
    python visualize_masac_trajectory_3d.py --model-dir <path> --render True
    python visualize_masac_trajectory_3d.py --max-steps 800 --save-path my_traj.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from copy import deepcopy

from xuance import get_runner
from xuance.environment import make_envs


def parse_args():
    pa = argparse.ArgumentParser("MaSAC 3D Trajectory Visualization")
    pa.add_argument("--model-dir", type=str,
                    default="results/masac/SimpleSpreadAviary3D/best_model/best_model.pth",
                    help="Path to trained MaSAC model (.pth or directory)")
    pa.add_argument("--device", type=str, default="cuda:0")
    pa.add_argument("--render", type=lambda x: x.lower() == "true", default=False,
                    help="Whether to render PyBullet GUI (True/False)")
    pa.add_argument("--max-steps", type=int, default=1600,
                    help="Maximum episode steps")
    pa.add_argument("--seed", type=int, default=1)
    pa.add_argument("--save-path", type=str, default="masac_trajectory_3d.png",
                    help="Output path for the trajectory plot image")
    return pa.parse_args()


def plot_trajectories(trajectories, lm_xyz, save_path, max_steps):
    """绘制 3 架无人机的 3D 轨迹图。"""
    colors = ["#e74c3c", "#2ecc71", "#3498db"]  # 红、绿、蓝
    labels = ["Drone 0", "Drone 1", "Drone 2"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, traj in enumerate(trajectories):
        traj = np.array(traj)  # (T, 3)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=colors[i], linewidth=1.5, label=labels[i])
        # 起始位置 — 圆形
        ax.scatter(*traj[0], color=colors[i], marker="o", s=80,
                   edgecolors="black", zorder=5)
        # 终止位置 — 星形
        ax.scatter(*traj[-1], color=colors[i], marker="*", s=150,
                   edgecolors="black", zorder=5)

    # 目标地标 — 菱形
    target_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, lm in enumerate(lm_xyz):
        ax.scatter(*lm, color=target_colors[i % len(target_colors)], marker="D", s=120,
                   edgecolors="black", linewidths=1.2, zorder=5,
                   label=f"Target {i}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"MaSAC — UAV 3D Trajectories (steps={max_steps})")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\n轨迹图已保存至 {save_path}")
    plt.show()


def main():
    args = parse_args()

    # ---- 1. 加载 MaSAC 智能体 ----
    print("[1/3] 加载 MaSAC 智能体 …")
    runner = get_runner(
        algo="masac",
        env="Drones",
        env_id="SimpleSpreadAviary3D",
        parser_args=Namespace(
            parallels=1,
            device=args.device,
            render=False,
            vectorize="DummyVecMultiAgentEnv",
        ),
    )
    agent = runner.agent
    # 直接加载 policy 权重，跳过 optimizer 恢复（MaSAC 的 optimizer 是 dict，
    # 框架 load_model 在恢复 optimizer 后会触发 AttributeError）
    import torch
    model_path = args.model_dir
    if model_path.endswith(".pth"):
        checkpoint = torch.load(model_path, map_location=args.device, weights_only=True)
    else:
        from pathlib import Path
        pth_files = list(Path(model_path).glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth file found in {model_path}")
        checkpoint = torch.load(str(pth_files[0]), map_location=args.device, weights_only=True)
    agent.policy.load_state_dict(checkpoint['policy'], strict=False)
    print(f"Successfully loaded model from '{model_path}'.")
    # 关闭训练环境，避免 PyBullet 冲突
    runner.envs.close()

    # ---- 2. 创建评估环境 ----
    print("[2/3] 创建评估环境 …")
    config_test = deepcopy(runner.config)
    config_test.parallels = 1
    config_test.render = args.render
    config_test.env_seed = args.seed
    config_test.max_episode_steps = args.max_steps
    config_test.vectorize = "DummyVecMultiAgentEnv"
    test_envs = make_envs(config_test)

    # ---- 3. 运行评估并记录轨迹 ----
    print("[3/3] 运行评估 episode …")
    N = 3  # 无人机数量
    obs_dict, info = test_envs.reset()  # obs_dict: List[dict], 长度=parallels=1

    # 获取底层 PyBullet 环境引用
    # test_envs.envs[0] = XuanCeMultiAgentEnvWrapper
    #   .env = Drones_MultiAgentEnv
    #     .env = SimpleSpreadAviary3D
    raw_env = test_envs.envs[0].env.env  # SimpleSpreadAviary3D 实例

    # 记录地标位置
    lm_xyz = raw_env.landmark_pos.copy()

    # 初始化轨迹记录
    trajectories = [[] for _ in range(N)]
    for i in range(N):
        pos = raw_env._getDroneStateVector(i)[0:3].copy()
        trajectories[i].append(pos)

    print(f"  无人机初始位置:")
    for i in range(N):
        print(f"    Drone {i}: {np.round(trajectories[i][0], 3).tolist()}")
    print(f"  地标位置:")
    for i, lm in enumerate(lm_xyz):
        print(f"    Target {i}: {np.round(lm, 3).tolist()}")

    # 评估循环
    rnn_hidden = agent.init_rnn_hidden(1) if agent.use_rnn else None
    for step in range(args.max_steps):
        # 智能体推理
        result = agent.action(obs_dict, rnn_hidden=rnn_hidden, test_mode=True)
        actions_dict = result["actions"]  # List[dict], 长度=parallels=1
        if agent.use_rnn:
            rnn_hidden = result["hidden_state"]

        # 环境步进
        obs_dict, rewards, terminated, truncated, info = test_envs.step(actions_dict)

        # 记录轨迹
        for i in range(N):
            pos = raw_env._getDroneStateVector(i)[0:3].copy()
            trajectories[i].append(pos)

        # 打印进度（每 200 步）
        if (step + 1) % 200 == 0:
            min_dists = []
            for lm in lm_xyz:
                dists = [np.linalg.norm(raw_env._getDroneStateVector(j)[0:3] - lm) for j in range(N)]
                min_dists.append(min(dists))
            print(f"  step {step+1:4d}/{args.max_steps} | "
                  f"landmark dists {[f'{d:.3f}' for d in min_dists]}")

        # 检查是否结束
        if truncated[0] if isinstance(truncated, (list, tuple)) else truncated:
            print(f"  Episode truncated at step {step+1}")
            break
        term_values = terminated[0] if isinstance(terminated, (list, tuple)) else terminated
        if isinstance(term_values, dict):
            if any(term_values.values()):
                print(f"  Episode terminated at step {step+1}")
                break
        elif term_values:
            print(f"  Episode terminated at step {step+1}")
            break

    # ---- 最终状态 ----
    print("\n最终位置:")
    for i in range(N):
        pos = raw_env._getDroneStateVector(i)[0:3]
        print(f"  Drone    {i}: {np.round(pos, 3).tolist()}")
    for i, lm in enumerate(lm_xyz):
        nearest = min(np.linalg.norm(raw_env._getDroneStateVector(j)[0:3] - lm) for j in range(N))
        print(f"  Target   {i}: {np.round(lm, 3).tolist()} (nearest drone: {nearest:.3f}m)")

    test_envs.close()

    # ---- 4. 绘制轨迹图 ----
    actual_steps = len(trajectories[0])
    plot_trajectories(trajectories, lm_xyz, args.save_path, actual_steps)
    print("Done.")


if __name__ == "__main__":
    main()
