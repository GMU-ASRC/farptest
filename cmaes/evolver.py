from typing import Callable, Literal

import cma
import numpy as np
from numpy.typing import NDArray


class VarConfig:
    def __init__(self, index_ref):
        """
        Expects a dictionary of the form [str : tuple], where the ith key is the name of the variable being controlled by the ith bounds
        and the tuple is the (min, max) bounds for the variable.
        """
        self.named_dict = index_ref
        self.from_unit_to_scaled = self.unit_unnormalize
        self.from_scaled_to_unit = self.unit_normalize

    @property
    def min_set(self):
        return [v[0] for v in self.named_dict.values()]

    @property
    def max_set(self):
        return [v[1] for v in self.named_dict.values()]

    @property
    def names(self):
        return list(self.named_dict.keys())

    def __len__(self):
        return len(self.named_dict)

    def unit_normalize(self, vector):
        return [self.map_to_unit(x, a, b) for x, a, b in zip(vector, self.min_set, self.max_set)]

    def unit_unnormalize(self, vector):
        return [self.map_from_unit(x, a, b) for x, a, b in zip(vector, self.min_set, self.max_set)]

    def as_dict(self):
        return self.named_dict

    def as_ordered_dict(self):
        return {"__order__": list(self.named_dict.keys()), **self.named_dict}

    @staticmethod
    def map_to_unit(x, in_a, in_b):
        return (x - in_a) / (in_b - in_a)

    @staticmethod
    def map_from_unit(x, out_a, out_b):
        return x * (out_b - out_a) + out_a

    @staticmethod
    def map_to_range(x, in_a, in_b, out_a, out_b):
        return (x - in_a) * (out_b - out_a) / (in_b - in_a) + out_a


class Evolver:
    def __init__(self,
        genome_size: int,
        target: float,
        var_bounds: dict[str, tuple[float, float]],
        seed: int = 20,
        pop_size: int = 10,
        max_iters: int = 10,
        center: list[float] | None = None,
        optim_dir: Literal["min", "max"] = "min",
        eval_mode: Literal["single", "batch"] = "batch",
    ):
        self.seed = seed
        assert optim_dir in ["min", "max"]
        self.optim_dir = optim_dir

        assert eval_mode in ["single", "batch"]
        self.eval_mode = eval_mode

        assert target >= 0, "Temporary limit; until I figure how to deal with negative targets for either `optim_dir`"

        self.options = {
            "bounds": [[0.0 for _ in range(genome_size)], [1.0 for _ in range(genome_size)]],
            "ftarget": -abs(target) if self.optim_dir == "max" else target,
            "maxiter": max_iters,
            "seed": self.seed,
            "popsize": pop_size,
        }
        print(self.options)
        assert genome_size == len(var_bounds)
        self.var_config_dict = var_bounds
        self.var_config = VarConfig(self.var_config_dict)
        x0 = []
        if center is None:
            x0 = [0.5 for _ in range(genome_size)]
        else:
            x0 = self.var_config.from_scaled_to_unit(center)

        self.es = cma.CMAEvolutionStrategy(x0=x0, sigma0=0.25, options=self.options)
        self.best_info = {}

    def on_close(self):
        print(self.best_info)

    def fitness_single(self, genome, iter_seed) -> float:
        raise NotImplementedError("You need to implement fitness_single")

    def fitness_batch(self, genomes, iter_seed) -> list[float]:
        raise NotImplementedError("You need to implement fitness_batch")

    def evolve(self, show_progress: bool = True) -> dict:
        try:
            self._evolve_helper(show_progress)
        except KeyboardInterrupt:
            print("Detected <C-c>; stopping now...")
        finally:
            self.on_close()

        return self.best_info

    def _evolve_helper(self, show_progress: bool):
        optim_multiplier = -1.0 if self.optim_dir == "max" else 1.0

        iters = 0
        while not self.es.stop():
            genomes = self.es.ask()
            fitnesses = []
            unnorm_genomes = [self.var_config.from_unit_to_scaled(genome) for genome in genomes]
            match self.eval_mode:
                case "single":
                    fitnesses = [
                        self.fitness_single(ung, self.seed + iters)
                        for ung in unnorm_genomes
                    ]

                case "batch":
                    fitnesses = self.fitness_batch(unnorm_genomes, self.seed + iters)

            self.es.tell(genomes, np.array(fitnesses) * optim_multiplier)
            if show_progress:
                self.es.disp()

            self.best_info = {
                "best_unnorm": self.var_config.from_unit_to_scaled(self.es.result.xbest),
                "best_fitness": self.es.result.fbest
            }
            iters += 1

        self.es.result_pretty()