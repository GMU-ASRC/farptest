"""
Find the best Homogeneous Agents for Milling
"""

import sys
import pathlib as pl
sys.path.append(str(pl.Path(__file__).parents[2]))

from ctypes import ArgumentError
from io import BytesIO
import argparse
import numpy as np
from tqdm import tqdm

from swarmsim.world.simulate import main as sim
from cmaes.CMAES import CMAES
from cmaes.farpcma import gene_to_world, DECISION_VARS

# NOTE: Legacy stuff from the novel-swarms era
SCALE = 1


def metric_to_canon(genome: tuple[float, float, float, float], body_length, scale=SCALE):
    v0, w0, v1, w1 = genome
    v0 *= scale / body_length
    v1 *= scale / body_length
    return (v0, w0, v1, w1)


def canon_to_metric(genome: tuple[float, float, float, float], body_length, scale=SCALE):
    v0, w0, v1, w1 = genome
    v0 /= scale / body_length
    v1 /= scale / body_length
    return (v0, w0, v1, w1)


def run(args, genome) -> float:

    # world_generator = get_world_generator(args.n, args.t)
    # world_config, *_ = world_generator(genome, [-1, -1, -1, -1])
    # note: world_config contains some persistent stuff like behaviors

    world_config = gene_to_world(args.n, genome, args.s)
    return sim(world_config=world_config, save_every_ith_frame=2, save_duration=1000, start_paused=True)


if __name__ == "__main__":
    """
    Example usage:
    `python -m demo.evolution.optim_milling.sim_results --v0 0.1531 --w0 0.3439 --v1 0.1485 --w1 0.1031 --n 10 --t 1000`
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=10, help="Number of agents")
    parser.add_argument("-t", type=int, default=1000, help="Environment Horizon")
    parser.add_argument("-s", type=int, default=2023, help="RNG seed")
    genome_parser = parser.add_mutually_exclusive_group(required=True)
    genome_parser.add_argument(
        "-g", "--genome",
        type=float,
        help="meters/second genome (4 floats expected: v0, w0, v1, w1)",
        default=None,
        nargs=4,
    )
    genome_parser.add_argument(
        "-ng", "--normalized_genome",
        type=float,
        help="Normalized genome values (4 floats expected between [0, 1]: v0, w0, v1, w1)",
        default=None,
        nargs=4,
    )

    args = parser.parse_args()

    genome = None
    if args.normalized_genome:
        genome = DECISION_VARS.from_unit_to_scaled(args.normalized_genome)

    elif args.genome:
        genome = args.genome

    assert genome is not None
    g = genome
    print(f"v0   (m/s):\t{g[0]:>16.12f}\tv1   (m/s):\t{g[2]:>16.12f}")
    print(f"w0 (rad/s):\t{g[1]:>16.12f}\tw1 (rad/s):\t{g[3]:>16.12f}")

    world = run(args, genome)
    name, value = world.metrics[0].name, world.metrics[0].value
    print(f"{name}: {value}")
