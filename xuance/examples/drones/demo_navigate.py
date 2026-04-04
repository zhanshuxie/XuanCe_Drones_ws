"""
单架无人机点到点导航 — PPO 训练/测试脚本

训练一架无人机从空间中随机起点 A 平稳飞行到随机终点 B。
环境: NavigateAviary (继承 BaseRLAviary, 速度控制)
算法: PPO (XuanCe)

配置: xuance/configs/ppo/Drone/NavigateAviary.yaml

用法:
    训练:       python demo_navigate.py --mode train
    基准训练:   python demo_navigate.py --mode benchmark
    可视化测试: python demo_navigate.py --mode test --render True --model-dir results/ppo/NavigateAviary/best_model/best_model.pth --test-episode 50 --parallels 1
    CPU:        python demo_navigate.py --device cpu
"""
import argparse
import numpy as np
from copy import deepcopy
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Single-UAV Point-to-Point Navigation with PPO")
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--env", type=str, default="Drone")
    parser.add_argument("--env-id", type=str, default="NavigateAviary")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallels", type=int, default=10)
    parser.add_argument("--render", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--test-episode", type=int, default=5)
    parser.add_argument("--mode", type=str, default="benchmark", choices=["train", "test", "benchmark"])
    parser.add_argument("--model-dir", type=str, default="models/ppo/")
    return parser.parse_args()


def run_test_with_success_count(runner, parser):
    """Custom test loop that counts successful episodes (terminated=True means reached target)."""
    from xuance.environment import make_envs

    config_test = deepcopy(runner.config)
    config_test.render = parser.render
    config_test.parallels = parser.parallels
    runner.agent.load_model(parser.model_dir)

    # Close training envs before opening test envs — pybullet only allows one GUI connection.
    if runner.envs is not None:
        runner.envs.close()
        runner.envs = None

    test_envs = make_envs(config_test)
    obs, infos = test_envs.reset()

    scores = []
    success_flags = []
    episode_score = np.zeros(parser.parallels)
    episode_success = np.zeros(parser.parallels, dtype=bool)
    current_episode = 0
    test_episodes = parser.test_episode

    print(f"\nRunning {test_episodes} test episodes...")

    while current_episode < test_episodes:
        runner.agent.obs_rms.update(obs)
        obs_proc = runner.agent._process_observation(obs)
        policy_out = runner.agent.action(obs_proc)
        next_obs, rewards, terminals, truncations, infos = test_envs.step(policy_out['actions'])

        episode_score += rewards

        for i in range(parser.parallels):
            if terminals[i]:
                episode_success[i] = True

        for i in range(parser.parallels):
            if terminals[i] or truncations[i]:
                scores.append(float(episode_score[i]))
                success_flags.append(bool(episode_success[i]))
                episode_score[i] = 0.0
                episode_success[i] = False
                obs[i] = infos[i]["reset_obs"]
                current_episode += 1
                status = "SUCCESS" if success_flags[-1] else "FAIL"
                print(f"  Episode {current_episode:3d}/{test_episodes}: score={scores[-1]:8.2f}  [{status}]")
                if current_episode >= test_episodes:
                    break

        obs = next_obs

    test_envs.close()

    n_success = sum(success_flags)
    print("\n---------------------Testing Results--------------------")
    print(f"Total Episodes : {test_episodes}")
    print(f"Success Count  : {n_success} / {test_episodes}")
    print(f"Success Rate   : {n_success / test_episodes * 100:.1f}%")
    print(f"Mean Score     : {np.mean(scores):.2f}  Std: {np.std(scores):.2f}")
    print(f"Best Score     : {max(scores):.2f}")
    print("--------------------------------------------------------")


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(algo=parser.algo,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser)

    if parser.mode == "test":
        run_test_with_success_count(runner, parser)
    else:
        run_kwargs = {}
        if parser.mode == "test" and parser.model_dir:
            run_kwargs['model_path'] = parser.model_dir
        runner.run(mode=parser.mode, **run_kwargs)
