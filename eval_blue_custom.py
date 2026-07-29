import os
import argparse
from pathlib import Path
from warnings import warn

import numpy as np
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
from swarmsim import config_from_yaml
from eval_genome import METRIC
from eval_genome import fitness_single as fitness_single_genome
from util import load_all_controllers, test_mp

cwd = Path(__file__).resolve().parent


def fitness_single(*args, **kwargs):
    load_all_controllers(cwd)
    return fitness_single_genome(*args, **kwargs)


def generate_configs(rng_seed=20, n=6, trials=100):
    seeds = np.random.default_rng(rng_seed).integers(
        0, 2**31, size=trials, dtype=np.int64
    )

    return [
        config_from_yaml(
            cwd / "world.yaml",
            m=METRIC,
            blue_controller='custom',
            # blue_controller_class=args.blue_controller,
            evader="pid",
            seed=seed,
            n=n,
        )
        for seed in seeds
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-b", "--blue_controller", type=str, default='CustomController',
    #     help="Path to blue controller",
    # )
    parser.add_argument(
        "-s", "--samples", type=int, default=100, help="Number of samples to test"
    )
    parser.add_argument(
        "-n", "--agents", type=int, default=6, help="Number of agents to test with"
    )
    parser.add_argument(
        "-r", "--rng_seed", type=int, default=20, help="Seed for random number generator"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    controller = 'CustomController'
    print(f"Testing controller: {controller} \twith {args.agents} agents")
    print(f"Base Seed: {args.rng_seed}")
    ns = args.samples

    _, rate = test_mp(generate_configs(
        rng_seed=args.rng_seed, n=args.agents, trials=args.samples),
        fitness_single,
    )
    print(f"{'Capture' if METRIC == 'ttc' else 'Detection'} rate:\t"
          f"{100 * rate:.2f}%\t({int(rate * ns)}/{ns})")
