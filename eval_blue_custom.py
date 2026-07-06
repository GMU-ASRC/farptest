import os
import argparse
from pathlib import Path
from warnings import warn

import numpy as np
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
from swarmsim import config_from_yaml
from swarmsim.util.processing.multicoreprocessing import process_map
from eval_genome import METRIC
from eval_genome import fitness_single as fitness_single_genome

cwd = Path(__file__).resolve().parent


def load_all_controllers(path=None, error='warn'):
    modules = {}
    for f in Path(cwd if path is None else path).glob('*Controller.py'):
        try:
            module, controller = load_cls_from_file(f, 'controller')
            modules[f] = (module, controller)
        except ImportError as err:
            if error == 'warn':
                warn(err.msg, stacklevel=1)
            elif error == 'raise':
                raise err
    return modules


def load_cls_from_file(path: os.PathLike | str, namespace: str):
    # only works if class has same name as file
    from importlib.util import spec_from_file_location, module_from_spec
    from swarmsim import register_dictlike_type

    f = Path(path)
    assert f.exists()

    spec = spec_from_file_location(f.stem, f)
    assert spec is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, module.__name__)
    register_dictlike_type(namespace, cls.__name__, cls)
    return module, cls


def fitness_single(*args, **kwargs):
    load_all_controllers()
    return fitness_single_genome(*args, **kwargs)


def test_mp(blue_controller=None, n=6, rng_seed=20, trials=100, tqdm_kwargs={}):
    seeds = np.random.default_rng(rng_seed).integers(
        0, 2**31, size=trials, dtype=np.int64
    )

    configs = [
        config_from_yaml(
            cwd / "world.yaml",
            m=METRIC,
            blue_controller='custom',
            # blue_controller_class=args.blue_controller,
            evader="pid",
            seed=seed,
            n=n,
        )
        for seed in seeds
    ]
    ret_arr = process_map(fitness_single, configs, **tqdm_kwargs)
    stats, successes = zip(*ret_arr)

    rate = 1 - sum(successes) / len(seeds)
    return stats, rate


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-b", "--blue_controller", type=str, default='CustomController',
    #     help="Path to blue controller",
    # )
    parser.add_argument(
        "-s", "--samples", type=int, default=100, help="Number of samples to test"
    )
    parser.add_argument(
        "-n", "--agents", type=int, default=6, help="Number of agents to test with"
    )
    parser.add_argument(
        "-r", "--rng_seed", type=int, default=20, help="Seed for random number generator"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    controller = 'CustomController'
    print(f"Testing controller: {controller} \twith {args.agents} agents")
    print(f"Base Seed: {args.rng_seed}")
    ns = args.samples

    # if args.blue_controller:
    #     load_all_controllers()
    # else:


    _, rate = test_mp(trials=args.samples,
                             rng_seed=args.rng_seed, n=args.agents)
    print(f"{'Capture' if METRIC == 'ttc' else 'Detection'} rate:\t"
          f"{100 * rate:.2f}%\t({int(rate * ns)}/{ns})")
