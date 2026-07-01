from eval_genome import *

if __name__ == "__main__":
    genome, args = parse_args()
    print(f"Testing genome: {genome} \twith {args.agents} agents")
    print(f"Base Seed: {args.rng_seed}")
    ns = args.samples

    _, rate = test_genome_mp(genome, trials=args.samples, rng_seed=args.rng_seed, n=args.agents)
    print(f"{'Capture' if METRIC == 'ttc' else 'Detection'} rate:\t"
          f"{100 * rate:.2f}%\t({int(rate * ns)}/{ns})")

