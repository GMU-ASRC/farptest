import sys
import pathlib as pl
sys.path.append(str(pl.Path(__file__).parents[1]))

from evolver import Evolver
from eval_genome import generate_configs, fitness_single
from util import test_mp


V_MAX, W_MAX = 0.3, 0.6

class FARPEvolver(Evolver):
    def __init__(self, seed=20):
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
        self.trials = 100
        self.n = 6

    def fitness_batch(self, genomes, iter_seed) -> list[float]:
        rates = []
        for genome in genomes:
            stats, rate = test_mp(
                generate_configs(genome, n=self.n, rng_seed=iter_seed, trials=self.trials),
                fitness_single
            )
            self.all_stats.extend(stats)
            rates.append(rate)

        return rates


if __name__ == "__main__":
    FARPEvolver(seed=10).evolve()