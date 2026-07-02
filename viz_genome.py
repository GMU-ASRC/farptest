import numpy as np
from eval_genome import fitness_single, cwd, parse_args, METRIC
from swarmsim import config_from_yaml

genome, args = parse_args()
print(f"Testing genome: {genome} \twith {args.agents} agents")
ns = args.samples

seeds = np.random.default_rng(args.rng_seed).integers(
    0, 2**31, size=ns, dtype=np.int64
)
print(f"Seeds: {seeds}")

configs = [
    config_from_yaml(
        cwd / "world.yaml",
        m=METRIC,
        evader="pid",
        g=genome,
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
