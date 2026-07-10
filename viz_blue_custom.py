import numpy as np
from eval_blue_custom import fitness_single, cwd, parse_args, METRIC, load_all_controllers
from swarmsim import config_from_yaml

args = parse_args()
load_all_controllers()
# print(f"Testing genome: {genome} \twith {args.agents} agents")
ns = args.samples

seeds = np.random.default_rng(args.rng_seed).integers(
    0, 2**31, size=ns, dtype=np.int64
)
print(f"Seeds: {seeds}")

configs = [
    config_from_yaml(
        cwd / "world.yaml",
        m=METRIC,
        blue_controller='custom',
        # blue_controller_class=args.blue_controller,
        evader="pid",
        seed=seed,
        n=args.agents,
    )
    for seed in seeds
]

successes = []
for c in configs:
    _, success = fitness_single(c, show_gui=True, start_paused=False)
    successes.append(success)

rate = 1 - sum(successes) / len(seeds)

print(f"{'Capture' if METRIC == 'ttc' else 'Detection'} rate:\t"
      f"{100 * rate:.2f}%\t({int(rate * ns)}/{ns})")