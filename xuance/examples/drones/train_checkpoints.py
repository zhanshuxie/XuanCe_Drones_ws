"""
训练checkpoint保存脚本 — MAPPO规划器 + PPO导航器

分别训练MAPPO高层规划器和PPO低层导航器各10M步，
每100k步保存一个checkpoint（共100个）+ 最优模型。

输出目录结构:
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

用法:
    python train_checkpoints.py --device cuda:0 --ckpt-root checkpoints
    python train_checkpoints.py --device cuda:0 --ckpt-root checkpoints --only planner
    python train_checkpoints.py --device cuda:0 --ckpt-root checkpoints --only navigator
"""
import os
import argparse
import numpy as np
from copy import deepcopy
from argparse import Namespace

from xuance import get_runner
from xuance.environment import make_envs


def parse_args():
    pa = argparse.ArgumentParser("Train checkpoints for hierarchical RL ablation")
    pa.add_argument("--device", type=str, default="cuda:0")
    pa.add_argument("--ckpt-root", type=str, default="checkpoints",
                    help="Root directory for saving checkpoints")
    pa.add_argument("--only", type=str, default=None, choices=["planner", "navigator"],
                    help="Only train one of the two models")
    return pa.parse_args()


def train_planner(device: str, ckpt_root: str):
    """训练MAPPO规划器，每100k步保存checkpoint。"""
    print("=" * 60)
    print("Phase A: Training MAPPO Planner (10M steps)")
    print("=" * 60)

    runner = get_runner(
        algo="mappo", env="DroneSpread", env_id="SimpleSpreadDrones",
        parser_args=Namespace(parallels=16, device=device),
    )
    agent = runner.agent
    n_envs = runner.n_envs  # 16
    running_steps = runner.config.running_steps  # 10_000_000
    eval_interval = runner.config.eval_interval  # 100_000
    test_episodes = runner.config.test_episode   # 5

    train_steps_per_epoch = max(1, eval_interval // n_envs)  # 6250
    num_epochs = running_steps // eval_interval  # 100

    # 创建测试环境
    config_test = deepcopy(runner.config)
    config_test.parallels = 1
    config_test.render = False
    test_envs = make_envs(config_test)

    planner_dir = os.path.join(ckpt_root, "planner")
    best_dir = os.path.join(planner_dir, "best")
    best_mean = -np.inf

    print(f"  n_envs={n_envs}, running_steps={running_steps}, "
          f"eval_interval={eval_interval}")
    print(f"  train_steps_per_epoch={train_steps_per_epoch}, "
          f"num_epochs={num_epochs}")
    print(f"  Checkpoints will be saved to: {os.path.abspath(planner_dir)}")
    print()

    for epoch in range(num_epochs):
        agent.train(train_steps=train_steps_per_epoch)

        # 测试
        test_scores = agent.test(
            test_episodes=test_episodes,
            test_envs=test_envs,
            close_envs=False,
        )
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)

        # 保存checkpoint
        step_k = (epoch + 1) * eval_interval // 1000
        ckpt_dir = os.path.join(planner_dir, f"{step_k}k")
        agent.save_model(model_name="model.pth", model_path=ckpt_dir)

        print(f"  Epoch {epoch + 1:3d}/{num_epochs} | "
              f"step={agent.current_step:>10d} | "
              f"score={mean_score:.3f} ± {std_score:.3f} | "
              f"saved → {step_k}k/model.pth")

        # 更新最优模型
        if mean_score > best_mean:
            best_mean = mean_score
            agent.save_model(model_name="model.pth", model_path=best_dir)
            print(f"  *** New best model! score={best_mean:.3f} ***")

    test_envs.close()
    runner.envs.close()
    print(f"\nPlanner training complete. Best score: {best_mean:.3f}")
    print(f"Checkpoints saved to: {os.path.abspath(planner_dir)}")
    print()


def train_navigator(device: str, ckpt_root: str):
    """训练PPO导航器，每100k步保存checkpoint（含obs_rms.npy）。"""
    print("=" * 60)
    print("Phase B: Training PPO Navigator (10M steps)")
    print("=" * 60)

    runner = get_runner(
        algo="ppo", env="Drone", env_id="NavigateAviary",
        parser_args=Namespace(
            parallels=10,
            device=device,
            render=False,
            running_steps=10_000_000,
            eval_interval=100_000,
        ),
    )
    agent = runner.agent
    n_envs = runner.n_envs  # 10
    running_steps = runner.config.running_steps  # 10_000_000
    eval_interval = runner.config.eval_interval  # 100_000
    test_episodes = runner.config.test_episode   # 5

    train_steps_per_epoch = max(1, eval_interval // n_envs)  # 10000
    num_epochs = running_steps // eval_interval  # 100

    # 创建测试环境
    config_test = deepcopy(runner.config)
    config_test.parallels = test_episodes
    config_test.render = False
    test_envs = make_envs(config_test)

    nav_dir = os.path.join(ckpt_root, "navigator")
    best_dir = os.path.join(nav_dir, "best")
    best_mean = -np.inf

    print(f"  n_envs={n_envs}, running_steps={running_steps}, "
          f"eval_interval={eval_interval}")
    print(f"  train_steps_per_epoch={train_steps_per_epoch}, "
          f"num_epochs={num_epochs}")
    print(f"  Checkpoints will be saved to: {os.path.abspath(nav_dir)}")
    print()

    for epoch in range(num_epochs):
        agent.train(train_steps=train_steps_per_epoch)

        # ���试
        test_scores = agent.test(
            test_episodes=test_episodes,
            test_envs=test_envs,
            close_envs=False,
        )
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)

        # 保存checkpoint（save_model自动保存obs_rms.npy）
        step_k = (epoch + 1) * eval_interval // 1000
        ckpt_dir = os.path.join(nav_dir, f"{step_k}k")
        agent.save_model(model_name="model.pth", model_path=ckpt_dir)

        print(f"  Epoch {epoch + 1:3d}/{num_epochs} | "
              f"step={agent.current_step:>10d} | "
              f"score={mean_score:.3f} ± {std_score:.3f} | "
              f"saved → {step_k}k/model.pth + obs_rms.npy")

        # 更新最优模型
        if mean_score > best_mean:
            best_mean = mean_score
            agent.save_model(model_name="model.pth", model_path=best_dir)
            print(f"  *** New best model! score={best_mean:.3f} ***")

    test_envs.close()
    runner.envs.close()
    print(f"\nNavigator training complete. Best score: {best_mean:.3f}")
    print(f"Checkpoints saved to: {os.path.abspath(nav_dir)}")
    print()


if __name__ == "__main__":
    args = parse_args()

    if args.only is None or args.only == "planner":
        train_planner(args.device, args.ckpt_root)

    if args.only is None or args.only == "navigator":
        train_navigator(args.device, args.ckpt_root)

    print("All training complete!")
