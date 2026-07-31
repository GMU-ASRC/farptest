import argparse
import datetime as dt
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import cma
import numpy as np
from swarmsim import config_from_yaml, run_sim, register_dictlike_type

from util import test_mp
from DiffuseSentinel import DiffuseSentinelController
from CMAES import CMAES
from OptimVar import CMAESVarSet

METRIC = 'ttd'
cwd = Path(__file__).resolve().parent.parent

VAR_CONFIGS = {
    "second_stage": [100, 300],
    "half_angle"  : [0.6, 2],
}
DECISION_VARS = CMAESVarSet(VAR_CONFIGS)
PERFECT_SCORE = -1.0

def gen_configs(genome, rng_seed, n, trials):
    seeds = np.random.default_rng(rng_seed).integers(
        0, 2**31, size=trials, dtype=np.int64
    )

    return [
        config_from_yaml(
            cwd / "world.yaml",
            m=METRIC,
            evader="pid",
            blue_controller="diff_sent",
            g=genome,
            seed=seed,
            n=n,
        )
        for seed in seeds
    ]


def fitness_single(
    config,
    show_gui=False,
    start_paused=False,
):
    world = run_sim(config, show_gui=show_gui, start_paused=start_paused)

    return world.metrics[0].value

def fitness_wrapper(genomes: list[list[float]], n: int, iter_seed: int, trials: int):
    all_stats, all_rates = [], []
    for genome in genomes:
        # stats, rate = test_genome_mp(genome, n=n, rng_seed=iter_seed)
        stats, rate = test_mp(gen_configs(
            genome, n=n, rng_seed=iter_seed, trials=trials),
            fitness_single,
        )
        all_stats.extend(stats)
        all_rates.append(-rate)

    return all_stats, all_rates


def optimize_w_cma(rng_seed: int, pop_size: int, max_iters: int, n: int, trials: int):
    iter_seeds = np.random.default_rng(rng_seed).integers(
        0, 2**31, size=max_iters, dtype=np.int64).tolist()

    bests = []
    cmaes = CMAES(
        fitness=fitness_wrapper,
        target=PERFECT_SCORE,
        seed=rng_seed,
        genome_size=len(VAR_CONFIGS),
        pop_size=pop_size,
        max_iters=max_iters,
    )
    try:
        best_norm_genome, best_fitness = cmaes.evolve(n, iter_seeds, trials)
        best_unnorm_genome = DECISION_VARS.from_unit_to_scaled(best_norm_genome)
        best = {
            "n": n,
            "unnorm_genome": best_unnorm_genome,
            "fitness": best_fitness,
        }
        print(best)

    except FileNotFoundError as fnfe:
        print(fnfe)
    except KeyboardInterrupt:
        print("Detected <C-c>; stopping now...")
    finally:
        dt_str = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"cmaes_{dt_str}.json", "w") as f:
            json.dump({
                "rng_seed": rng_seed,
                "pop_size": pop_size,
                "max_iters": max_iters,
                "var_configs": VAR_CONFIGS,
                "bests": bests,
                "runs": cmaes.all_run_stats,
            }, f)


if __name__ == "__main__":
    register_dictlike_type("controller", "DiffuseSentinelController", DiffuseSentinelController)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=6, help="Number of defending agents")
    parser.add_argument("-rs", "--rng_seed", type=int, default=402, help="Seed for RNG")
    parser.add_argument("-t", "--trials", type=int, default=1000, help="Number of trials")
    parser.add_argument("-p", "--pop_size", type=int, default=10, help="Population size")
    parser.add_argument("-mi", "--max_iters", type=int, default=100, help="Maximum number of iterations")

    args = parser.parse_args()
    print("CMA-ES run info:")
    print(f"\tn         = {args.n}")
    print(f"\trng_seed  = {args.rng_seed}")
    print(f"\ttrials    = {args.trials}")
    print(f"\tpop_size  = {args.pop_size}")
    print(f"\tmax_iters = {args.max_iters}\n")

    optimize_w_cma(
        rng_seed=args.rng_seed,
        pop_size=args.pop_size,
        max_iters=args.max_iters,
        n=args.n,
        trials=args.trials
    )
