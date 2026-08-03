import datetime as dt
import json
import os
import sys
import time
import pathlib as pl
sys.path.append(str(pl.Path(__file__).parents[1]))

import pandas as pd
from evolver import Evolver
from eval_genome import generate_configs, fitness_single
from util import test_mp


V_MAX, W_MAX = 0.3, 0.6

class FARPEvolver(Evolver):
    def __init__(self, n, trials, seed=20):
        super().__init__(
            genome_size=4,
            target=1,
            var_bounds={
                "v0": (-V_MAX, V_MAX),
                "w0": (-W_MAX, W_MAX),  # Radians / second
                "v1": (-V_MAX, V_MAX),
                "w1": (-W_MAX, W_MAX),  # Radians / second
            },
            seed=seed,
            pop_size=15,
            max_iters=50,
            center=[0.2, 0.2, 0.2, -0.2],
            optim_dir="max",
            eval_mode="batch",
        )
        self.all_stats = []
        self.n = n
        self.trials = trials
        self.start_time = time.time()

    def fitness_batch(self, genomes, iter_seed) -> list[float]:
        rates = []
        for genome in genomes:
            stats, rate = test_mp(
                generate_configs(
                    genome, n=self.n, rng_seed=self.seed, trials=self.trials
                ),
                fitness_single
            )
            self.all_stats.append({
                "genome": genome,
                "n": self.n,
                "iter_seed": self.seed,
                "trials": self.trials,
                "runs": stats
            })
            rates.append(rate)

        return rates

    def on_close(self):
        os.makedirs("kcm_results", exist_ok=True)
        with open(f"kcm_results/kcm_n_{self.n}.json", "w") as f:
            json.dump({
                "train_time": time.time() - self.start_time,
                "rng_seed": self.seed,
                "cmaes_options": self.options,
                "var_configs": self.var_config_dict,
                "bests": self.best_info,
                "all_stats": self.all_stats,
            }, f, indent=4)


def eval_controller_over_n(best_of_each_n_results):
    best_each_n = pd.DataFrame(best_of_each_n_results)
    print(best_each_n)

    seed, trials = 32, 100
    n_range = [*range(3, 25+1)]

    def test_genome_over_n(genome) -> list[float]:
        success_trajectory = []
        for n in n_range:
            _, rate = test_mp(generate_configs(
                genome, rng_seed=seed, n=n, trials=trials),
                fitness_single,
            )
            success_trajectory.append(rate)

        return success_trajectory

    control_of = [4, 5, 6, 7, 8, 10, 15]
    genomes = best_each_n[best_each_n["n"].isin(control_of)]["best_unnorm"].tolist()

    results = {}
    for n, genome in zip(control_of, genomes):
        results[f"c{n}"] = test_genome_over_n(genome)

    os.makedirs("kcm_results", exist_ok=True)
    with open("kcm_results/eval_over_n.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # PART 1
    TRIALS = 10
    best_of_each_n_results = []
    for n in range(1, 40+1):
        fe = FARPEvolver(n=n, trials=TRIALS, seed=20)
        fe.evolve()
        best_of_each_n_results.append({
            "n": n,
            **fe.best_info,
        })

    os.makedirs("kcm_results", exist_ok=True)
    with open("kcm_results/best_of_each_n.json", "w") as f:
        json.dump(best_of_each_n_results, f, indent=4)

    # PART 2
    eval_controller_over_n(best_of_each_n_results)