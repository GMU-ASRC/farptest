import numpy as np
from eval_red_custom import fitness_single, cwd, parse_args, METRIC, load_all_controllers, generate_configs

args = parse_args()
load_all_controllers()
# print(f"Testing genome: {genome} \twith {args.agents} agents")
ns = args.samples

configs = generate_configs(trials=ns, rng_seed=args.rng_seed, n=args.agents)

successes = []
for c in configs:
    _, success = fitness_single(c, show_gui=True, start_paused=False)
    successes.append(success)

rate = 1 - sum(successes) / len(configs)

print(f"{'Capture' if METRIC == 'ttc' else 'Detection'} rate:\t"
      f"{100 * rate:.2f}%\t({int(rate * ns)}/{ns})")
