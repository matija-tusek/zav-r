import argparse
import json
from main.fitness_evaluator import replay_checkpoints

def load_genome_from_json(filename: str):
    """Load genome from a JSON file."""
    with open("best_creature.json", 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Replay saved creature checkpoints.")
    parser.add_argument(
        "--genome",
        type=str,
        default="best_creature.json",
        help="Path to the genome JSON file."
    )
    args = parser.parse_args()

    # Load the genome
    genome = load_genome_from_json(args.genome)

    # Replay checkpoints
    print(f"Replaying checkpoints for genome: {args.genome}")
    replay_checkpoints(genome)

if __name__ == "__main__":
    main()