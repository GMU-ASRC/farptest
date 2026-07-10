import os
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
from swarmsim import config_from_yaml, run_sim
from swarmsim.util.processing.multicoreprocessing import process_map

cwd = Path(__file__).resolve().parent

METRIC = 'ttd'


def parse_list_arg(strings: list[str]) -> list[float]:
    from ast import literal_eval
    from numbers import Number
    splits = []
    for s in strings:
        splits.extend(s.split(','))
    elems = [s.strip().strip('"\',').lstrip('([').rstrip(')]').strip() for s in splits]
    l = [literal_eval(el) for el in elems if el]
    if not all(isinstance(el, Number) for el in l):
        raise ValueError("All elements must be numbers.")
    return l


def fitness_single(
    config,
    show_gui=False,
    start_paused=False,
):
    world = run_sim(config, show_gui=show_gui, start_paused=start_paused)

    out = world.metrics[0].value
    stat = Counter()
    for m in world.metrics:
        stat[m.name] += m.value

    return stat, out


def test_genome_mp(genome, n=6, rng_seed=20, trials=100, tqdm_kwargs={}):
    seeds = np.random.default_rng(rng_seed).integers(
        0, 2**31, size=trials, dtype=np.int64
    )

    configs = [
        config_from_yaml(
            cwd / "world.yaml",
            m=METRIC,
            evader="pid",
            g=genome,
            seed=seed,
            n=n,
        )
        for seed in seeds
    ]
    ret_arr = process_map(fitness_single, configs, **tqdm_kwargs)
    stats, successes = zip(*ret_arr)

    rate = 1 - sum(successes) / len(seeds)
    return stats, rate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("genome",
        nargs='+', help="Genome values as a list", type=str, action="extend",
    )
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

    genome = parse_list_arg(args.genome)
    return genome, args


if __name__ == "__main__":
    genome, args = parse_args()
    print(f"Testing genome: {genome} \twith {args.agents} agents")
    print(f"Base Seed: {args.rng_seed}")
    ns = args.samples

    _, rate = test_genome_mp(genome, trials=args.samples,
                             rng_seed=args.rng_seed, n=args.agents)
    print(f"{'Capture' if METRIC == 'ttc' else 'Detection'} rate:\t"
          f"{100 * rate:.2f}%\t({int(rate * ns)}/{ns})")
