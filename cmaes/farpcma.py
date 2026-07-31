import sys
import pathlib as pl
sys.path.append(str(pl.Path(__file__).parents[1]))

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from CMAES import CMAES
from OptimVar import CMAESVarSet

from eval_genome import generate_configs, fitness_single
from util import test_mp


cwd = Path(__file__).resolve().parent
# arbitrary linear and angular velocity values; can be changed
V_MAX, W_MAX = 0.3, 0.6
# NOTE: Perfect score implies 100% (0.0) failure rate; negating here to
# turn the success rate maximization problem into a minimization one for CMA-ES
PERFECT_SCORE = -1.0
VAR_CONFIGS = {
    "forward_rate_0": [-V_MAX, V_MAX],
    "turning_rate_0": [-W_MAX, W_MAX],  # Radians / second
    "forward_rate_1": [-V_MAX, V_MAX],
    "turning_rate_2": [-W_MAX, W_MAX],  # Radians / second
}
DECISION_VARS = CMAESVarSet(VAR_CONFIGS)


def fitness_wrapper(genomes: list[list[float]], n: int, iter_seed: int, trials: int):
    all_stats, all_rates = [], []
    for genome in genomes:
        # stats, rate = test_genome_mp(genome, n=n, rng_seed=iter_seed)
        stats, rate = test_mp(generate_configs(
            genome, n=n, rng_seed=iter_seed, trials=trials),
            fitness_single,
        )
        all_stats.extend(stats)
        all_rates.append(-rate)

    return all_stats, all_rates


def find_cma(
    rng_seed: int,
    pop_size: int,
    max_iters: int,
    trials: int,
    n_range: list[int] = [],
):
    iter_seeds = np.random.default_rng(rng_seed).integers(
        0, 2**31, size=max_iters, dtype=np.int64)

    bests, all_combined_stats = [], []
    try:
        for n in n_range:
            cmaes = CMAES(
                fitness=fitness_wrapper,
                target=PERFECT_SCORE,
                seed=rng_seed,
                genome_size=4,
                pop_size=pop_size,
                max_iters=max_iters,
            )
            print(f"\n\n==================== n = {n} ====================\n\n")
            best_norm_genome, best_fitness = cmaes.evolve(n, iter_seeds, trials)
            best_unnorm_genome = DECISION_VARS.from_unit_to_scaled(best_norm_genome)
            best_this_n = {
                "n": n,
                "unnorm_genome": best_unnorm_genome,
                "fitness": best_fitness,
            }
            all_combined_stats.extend(cmaes.all_run_stats)
            print(best_this_n)
            bests.append(best_this_n)

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
                "runs": all_combined_stats,
            }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CMAES-specific config
    parser.add_argument(
        "-n",
        "--n_range",
        type=int,
        nargs=3,
        required=True,
        help="-n <min> <max> <step>; Number of defending agents (interval inclusive)",
    )
    parser.add_argument("-rs", "--rng_seed", type=int, default=40, help="Seed for RNG")
    parser.add_argument(
        "-t", "--trials", type=int, default=100, help="Number of trials"
    )
    parser.add_argument(
        "-p", "--pop_size", type=int, default=50, help="Population size"
    )
    parser.add_argument(
        "-mi",
        "--max_iters",
        type=int,
        required=True,
        help="Maximum number of iterations",
    )

    args = parser.parse_args()
    start = time.time()
    

    # Print helpful info
    min_, max_, step = args.n_range
    list_of_ns = list(range(min_, max_+1, step))
    print("CMA-ES run info:")
    print(f"\tn         = range({args.n_range})")
    print(f"\trng_seed  = {args.rng_seed}")
    print(f"\ttrials    = {args.trials}")
    print(f"\tpop_size  = {args.pop_size}")
    print(f"\tmax_iters = {args.max_iters}")

    find_cma(
        rng_seed=args.rng_seed,
        pop_size=args.pop_size,
        max_iters=args.max_iters,
        trials=args.trials,
        n_range=list_of_ns,
    )
    print(f"Took {time.time() - start} seconds")
