"""
Flat MARL Baseline Benchmark — 3D SimpleSpreadAviary

3 drones directly controlled by MARL algorithms in 3D space to cover 3 targets.
No hierarchy. 1600 steps per episode. Reward: MPE simple_spread style.
Dynamics: PyBullet CF2X physics.

Supported algorithms: MAPPO, MADDPG, MASAC, MATD3
Configs: xuance/configs/<algo>/Drones/SimpleSpreadAviary3D.yaml

Usage:
    python benchmark_flat_3d.py --algo mappo  --mode benchmark
    python benchmark_flat_3d.py --algo maddpg --mode benchmark
    python benchmark_flat_3d.py --algo masac  --mode benchmark
    python benchmark_flat_3d.py --algo matd3  --mode benchmark

    python benchmark_flat_3d.py --algo mappo --mode train
    python benchmark_flat_3d.py --algo mappo --mode test --model-dir <path>
"""
import argparse
from argparse import Namespace
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Flat MARL 3D Benchmark")
    parser.add_argument("--algo", type=str, default="mappo",
                        help="Algorithm: mappo, maddpg, masac, matd3")
    parser.add_argument("--env", type=str, default="Drones")
    parser.add_argument("--env-id", type=str, default="SimpleSpreadAviary3D")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallels", type=int, default=16)
    parser.add_argument("--test-episode", type=int, default=10)
    parser.add_argument("--test-parallels", type=int, default=5,
                        help="Number of parallel envs for evaluation (default: same as test-episode)")
    parser.add_argument("--mode", type=str, default="benchmark",
                        choices=["train", "test", "benchmark"])
    parser.add_argument("--model-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == "test":
        args.parallels = 1
        args.vectorize = "DummyVecMultiAgentEnv"
    clean_args = Namespace(**{k: v for k, v in vars(args).items() if v is not None})
    runner = get_runner(algo=args.algo,
                        env=args.env,
                        env_id=args.env_id,
                        parser_args=clean_args)
    run_kwargs = {}
    if args.model_dir is not None:
        run_kwargs['model_path'] = args.model_dir
    if args.mode == "benchmark":
        test_par = args.test_parallels if args.test_parallels is not None else args.test_episode
        run_kwargs['test_parallels'] = test_par
        run_kwargs['test_episodes'] = args.test_episode
    runner.run(mode=args.mode, **run_kwargs)
