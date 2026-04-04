"""
Flat MARL Baseline Benchmark — 3D SimpleSpreadDrones

3 drones directly controlled by MARL algorithms in 3D space to cover 3 targets.
No hierarchy. 1600 steps per episode. Reward: MPE simple_spread style.
Dynamics: MPE integrate_state (force-based, with damping).

Supported algorithms: MAPPO, MADDPG, MASAC, MATD3
Configs: xuance/configs/<algo>/DroneSpread3D/SimpleSpreadDrones3D.yaml

Usage:
    python benchmark_flat_3d.py --algo mappo  --mode benchmark
    python benchmark_flat_3d.py --algo maddpg --mode benchmark
    python benchmark_flat_3d.py --algo masac  --mode benchmark
    python benchmark_flat_3d.py --algo matd3  --mode benchmark

    python benchmark_flat_3d.py --algo mappo --mode train
    python benchmark_flat_3d.py --algo mappo --mode test --model-dir <path>
"""
import argparse
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Flat MARL 3D Benchmark")
    parser.add_argument("--algo", type=str, default="mappo",
                        help="Algorithm: mappo, maddpg, masac, matd3")
    parser.add_argument("--env", type=str, default="DroneSpread3D")
    parser.add_argument("--env-id", type=str, default="SimpleSpreadDrones3D")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallels", type=int, default=16)
    parser.add_argument("--test-episode", type=int, default=5)
    parser.add_argument("--mode", type=str, default="benchmark",
                        choices=["train", "test", "benchmark"])
    parser.add_argument("--model-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == "test":
        args.parallels = 1
        args.vectorize = "DummyVecMultiAgentEnv"
    runner = get_runner(algo=args.algo,
                        env=args.env,
                        env_id=args.env_id,
                        parser_args=args)
    run_kwargs = {}
    if args.model_dir is not None:
        run_kwargs['model_path'] = args.model_dir
    runner.run(mode=args.mode, **run_kwargs)
