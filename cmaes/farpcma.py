import argparse
import datetime as dt
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from CMAES import CMAES
from numpy.typing import NDArray
from OptimVar import CMAESVarSet
from swarmsim.util.processing.multicoreprocessing import process_map
from swarmsim.world import config_from_yaml
from swarmsim.world.simulate import main as simulate

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


def gene_to_world(
    n: int, unnorm_genome: list[float], seed: int, yaml_conf: dict[str, str] = {}
):
    kwargs = {
        "n": n,
        "seed": seed,
        "g": unnorm_genome,
    }
    for key in ["m", "evader", "defender"]:
        if key in yaml_conf and yaml_conf[key] is not None:
            kwargs[key] = yaml_conf[key]

    return config_from_yaml(cwd / "world.yaml", **kwargs)


def fitness_single(config: tuple[int, NDArray[np.float64], int, dict[str, str]]):
    n, norm_genome, seed, yaml_conf = config
    assert len(norm_genome.shape) == 1

    unnorm_genome = DECISION_VARS.from_unit_to_scaled(norm_genome)
    world_conf = gene_to_world(n, unnorm_genome, seed, yaml_conf)
    world = simulate(world_conf, show_gui=False, start_paused=False)

    out = world.metrics[0].value

    stat = Counter()
    n = 0
    for agent in world.population:
        if agent.team == "blue":
            n += 1

    stat["n"] = n
    stat["seed"] = int(seed)
    stat["unnorm_genome"] = unnorm_genome
    for m in world.metrics:
        stat[m.name] += m.value

    return stat, out


def fitness_mp(
    n: int,
    norm_genomes: NDArray[np.float64],
    seeds: list[int],
    yaml_conf: dict[str, str],
):
    succ_rates = []
    all_stats = []
    for norm_genome in norm_genomes:
        configs = [(n, norm_genome, seed, yaml_conf) for seed in seeds]
        ret_arr = process_map(fitness_single, configs)
        stats, successes = zip(*ret_arr)

        all_stats.extend(stats)
        succ_rates.append(sum(successes) / len(seeds))

    return all_stats, succ_rates

def find_cma(
    n_range: list[int],
    rng_seed: int,
    trial_seeds_count: int,
    pop_size: int,
    max_iters: int,
    yaml_conf: dict[str, str],
):
    trial_seeds = np.random.default_rng(rng_seed).integers(
        0, 2**31, size=trial_seeds_count, dtype=np.int64
    )
    assert isinstance(n_range, list)

    cmaes = CMAES(
        fitness=fitness_mp,
        target=PERFECT_SCORE,
        seed=rng_seed,
        genome_size=4,
        pop_size=pop_size,
        max_iters=max_iters,
    )
    bests = []
    try:
        for n in n_range:
            best_norm_genome, best_fitness = cmaes.evolve(n, trial_seeds, yaml_conf)
            best_unnorm_genome = DECISION_VARS.from_unit_to_scaled(best_norm_genome)
            bests.append(
                {
                    "n": n,
                    "unnorm_genome": best_unnorm_genome,
                    "fitness": best_fitness,
                }
            )

    except FileNotFoundError as fnfe:
        print(fnfe)
    except KeyboardInterrupt:
        print("Detected <C-c>; stopping now...")
    finally:
        cmaes.es.result_pretty()

        dt_str = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"results/cmaes_{dt_str}.json", "w") as f:
            json.dump({
                "rng_seed": rng_seed,
                "pop_size": pop_size,
                "max_iters": max_iters,
                "var_configs": VAR_CONFIGS,
                "bests": bests,
                "runs": cmaes.all_run_stats,
            }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CMAES-specific config
    parser.add_argument(
        "-n",
        "--n_range",
        type=int,
        nargs="+",
        required=True,
        help="Number of defending agents",
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

    # YAML-specific config
    # TODO: add validation before starting CMA-ES run
    parser.add_argument("-m", "--metricfile", type=str, default=None, help="Metric file")
    parser.add_argument("-d", "--defender", type=str, default=None, help="Defender controller")
    parser.add_argument("-e", "--evader", type=str, default=None, help="Evader controller")

    args = parser.parse_args()
    yaml_conf: dict[str, str] = {
        "m": args.metricfile,
        "defender": args.defender,
        "evader": args.evader,
    }

    start = time.time()
    find_cma(
        n_range=args.n_range,
        rng_seed=args.rng_seed,
        trial_seeds_count=args.trials,
        pop_size=args.pop_size,
        max_iters=args.max_iters,
        yaml_conf=yaml_conf,
    )
    print(f"Took {time.time() - start} seconds")
