import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import cma
import torch
from torch.nn.utils import parameters_to_vector
from cmaes.evolver import Evolver
from CustomController import FarpRNN, CustomController
from eval_genome import fitness_single
from swarmsim import config_from_yaml, register_dictlike_type
from util import test_mp

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


V_MAX, W_MAX = 0.3, 0.6
CWD = Path(__file__).resolve().parent


class FARPNNDefenseEvolver(Evolver):
    def __init__(self, seed=20):
        sample_model = FarpRNN()
        x0 = parameters_to_vector(sample_model.parameters()).numpy()

        super().__init__(
            genome_size=len(x0),
            target=1,
            var_bounds={
                "v0": (-V_MAX, V_MAX),
                "w0": (-W_MAX, W_MAX),
                "v1": (-V_MAX, V_MAX),
                "w1": (-W_MAX, W_MAX),
            },
            center=x0,
            seed=seed,
            pop_size=15,
            max_iters=50,
            optim_dir="max",
            eval_mode="batch",
            cmaes_options = {
                "tolfunhist": 1e-3,
                "tolfun": 1e-3,
                "tolx": 1e-6,
                "tolflatfitness": 5,
            }
        )
        self.all_stats = []
        self.n = 6
        self.trials = 100
        self.start_time = time.time()

    def gen_configs(self, genome, iter_seed):
        seeds = np.random.default_rng(iter_seed).integers(
            0, 2**31, size=self.trials, dtype=np.int64
        )

        return [
            config_from_yaml(
                CWD / "world.yaml",
                m="ttd",
                blue_controller='farpnn',
                evader="pid",
                seed=seed,
                g=genome,
                n=self.n,
            )
            for seed in seeds
        ]

    def fitness_batch(self, genomes, iter_seed) -> list[float]:
        rates = []
        for genome in genomes:
            stats, rate = test_mp(
                self.gen_configs(genome, iter_seed),
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
        os.makedirs("fnn_results", exist_ok=True)
        with open(f"fnn_results/fnn_results.json", "w") as f:
            json.dump({
                "train_time": time.time() - self.start_time,
                "rng_seed": self.seed,
                "cmaes_options": self.options,
                "var_configs": self.var_config_dict,
                "bests": self.best_info,
                "all_stats": self.all_stats,
            }, f, indent=4)


if __name__ == "__main__":
    register_dictlike_type("controller", "CustomController", CustomController)
    with torch.inference_mode():
        FARPNNDefenseEvolver(seed=20).evolve()