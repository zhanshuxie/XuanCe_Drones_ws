"""
三维高层规划器训练脚本 — MAPPO on SimpleSpreadDrones3DPlanner

在纯 3D 粒子环境中训练 3-agent 合作覆盖任务（simple_spread）。
训练完成后可与低层 Navigate PPO 模型拼接部署（见 demo_hierarchical_deploy_3d.py）。

配置: xuance/configs/mappo/DroneSpreadPlanner3D/SimpleSpreadDrones3DPlanner.yaml

用法:
    训练:       python demo_simple_spread_planner_3d.py --mode train
    基准训练:   python demo_simple_spread_planner_3d.py --mode benchmark
    可视化测试: python demo_simple_spread_planner_3d.py --mode test --model-dir results/mappo/SimpleSpreadDrones3DPlanner/best_model/best_model.pth --device cpu --test-episode 1
"""
import os
import argparse
import numpy as np
from copy import deepcopy
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("High-level 3D Planner: MAPPO on SimpleSpreadDrones3DPlanner")
    parser.add_argument("--algo", type=str, default="mappo")
    parser.add_argument("--env", type=str, default="DroneSpreadPlanner3D")
    parser.add_argument("--env-id", type=str, default="SimpleSpreadDrones3DPlanner")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallels", type=int, default=16)
    parser.add_argument("--test-episode", type=int, default=5)
    parser.add_argument("--mode", type=str, default="benchmark",
                        choices=["train", "test", "benchmark"])
    parser.add_argument("--model-dir", type=str, default="models/mappo/")
    parser.add_argument("--gif-path", type=str, default="test_result.gif",
                        help="Output path for the GIF (only used in --mode test)")
    return parser.parse_args()


def save_gif(runner, model_path, gif_path, n_episodes=1):
    """Run test episodes manually and save a GIF directly to gif_path."""
    import imageio
    from xuance.environment import make_envs

    agent = runner.agent
    agent.load_model(model_path)

    config_test = deepcopy(runner.config)
    config_test.parallels = 1
    config_test.render = False
    config_test.render_mode = "rgb_array"
    config_test.vectorize = "DummyVecMultiAgentEnv"
    envs = make_envs(config_test)

    frames = []
    scores = []

    for ep in range(n_episodes):
        obs_dict, _ = envs.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # Render frame from the raw environment
            raw_env = envs.envs[0] if hasattr(envs, 'envs') else envs
            frame = raw_env.render()
            if frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3:
                frames.append(frame.astype(np.uint8))

            # Get actions (test mode: no exploration)
            avail_actions = envs.buf_avail_actions if agent.use_actions_mask else None
            state = envs.buf_state if agent.use_global_state else None
            rnn_hidden_actor, rnn_hidden_critic = agent.init_rnn_hidden(1)
            policy_out = agent.action(
                obs_dict=obs_dict,
                state=state,
                avail_actions_dict=avail_actions,
                rnn_hidden_actor=rnn_hidden_actor,
                rnn_hidden_critic=rnn_hidden_critic,
                test_mode=True,
            )
            actions_dict = policy_out['actions']

            obs_dict, rewards_dict, terminated_dict, truncated, _ = envs.step(actions_dict)

            ep_reward += float(np.mean(list(rewards_dict[0].values())))
            done = all(terminated_dict[0].values()) or truncated[0]

        scores.append(ep_reward)
        print(f"  Episode {ep + 1}: score = {ep_reward:.3f}")

    envs.close()

    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=10, loop=0)
        print(f"GIF saved to: {os.path.abspath(gif_path)}")
    else:
        print("No frames captured — GIF not saved.")

    return scores


if __name__ == '__main__':
    parser = parse_args()
    if parser.mode == "test":
        parser.parallels = 1
        parser.vectorize = "DummyVecMultiAgentEnv"
    runner = get_runner(algo=parser.algo,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser)
    run_kwargs = {}
    if parser.model_dir is not None:
        run_kwargs['model_path'] = parser.model_dir

    if parser.mode == "test":
        # Standard score evaluation (via framework)
        runner.run(mode=parser.mode, **run_kwargs)
        # Additionally save a GIF directly to a local path (bypasses temp-file permission issues)
        print("\nSaving GIF...")
        save_gif(runner, parser.model_dir, parser.gif_path, n_episodes=1)
    else:
        runner.run(mode=parser.mode, **run_kwargs)
