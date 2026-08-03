import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
import datetime as dt
import json

import numpy as np
from DiffuseSentinel import DiffuseSentinelController
from swarmsim import config_from_yaml
from evolver import Evolver
from eval_genome import fitness_single
from util import test_mp

CWD = Path(__file__).resolve().parent.parent
METRIC = "ttd"

class SentinelEvolver(Evolver):
    def __init__(self, seed=20):
        super().__init__(
            genome_size=2,
            target=1,
            var_bounds={
                "second_stage": (100, 300),
                "half_angle"  : (0.6, 2),
            },
            seed=seed,
            pop_size=10,
            max_iters=50,
            center=[200, 1.4],
            optim_dir="max",
            eval_mode="batch"
        )
        self.all_stats = []

    def gen_configs(self, genome, rng_seed, n, trials):
        seeds = np.random.default_rng(rng_seed).integers(
            0, 2**31, size=trials, dtype=np.int64
        )

        return [
            config_from_yaml(
                CWD / "world.yaml",
                m=METRIC,
                evader="pid",
                blue_controller="diff_sent",
                g=genome,
                seed=seed,
                n=n,
            )
            for seed in seeds
        ]

    def fitness_batch(self, genomes: list[list[float]], iter_seed):
        rates = []
        for genome in genomes:
            stats, rate = test_mp(self.gen_configs(
                genome, n=6, rng_seed=iter_seed, trials=100),
                fitness_single,
            )
            self.all_stats.extend(stats)
            rates.append(rate)

        return rates

    def on_close(self):
        print(self.best_info)

        now = dt.datetime.now()
        now_str = now.strftime("%Y-%h-%d_%H-%M-%S")
        with open(f"js_{now_str}.json", "w") as f:
            json.dump({
                **self.best_info,
            }, f)


if __name__ == "__main__":
    register_dictlike_type("controller", "DiffuseSentinelController", DiffuseSentinelController)
    SentinelEvolver().evolve()