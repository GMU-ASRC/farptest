from evotorch.logging import PandasLogger
import json

import torch
from torch.nn.utils import parameters_to_vector
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger

from swarmsim import register_dictlike_type
from eval_blue_custom import generate_configs, fitness_single
from util import test_mp
from CustomController import FarpRNN


# 2. Define the Fitness Evaluation Function
def evaluate_network(fnn: FarpRNN) -> float:
    """
    Evaluates the neural network on a synthetic dataset.
    Returns a scalar fitness score (higher is better).
    """
    
    _, rate = test_mp(
        generate_configs(
            rng_seed=torch.seed(), n=6, trials=100,
            blue_controller_class="CustomController",
            g=parameters_to_vector(fnn.parameters()).numpy()
        ),
        fitness_single,
    )
    
    return rate


def main():
    problem = NEProblem(
        objective_sense="max",
        network=FarpRNN,
        network_eval_func=evaluate_network
    )

    # 4. Instantiate the CMA-ES Searcher
    searcher = CMAES(
        problem,
        stdev_init=1.0,
        popsize=3,
    )

    # Attach a logger to print progress every 10 generations
    logger = StdOutLogger(searcher, interval=1)
    pd_logger = PandasLogger(searcher)

    # 5. Run the Evolutionary Optimization
    print("Starting CMA-ES neuroevolution search...")
    searcher.run(150)

    pd_logger.to_dataframe().to_csv("test.csv")

    # 6. Extract the Best Evolved Network
    # Retrieve the best solution found throughout the run
    best_individual = searcher.status["best"]
    print(best_individual)


if __name__ == "__main__":
    with torch.inference_mode():
        main()