import json
from fitness_evaluator import get_fitness_score
from replay import replay_checkpoints

GENOME_PATH = "best_creature.json"
TRAIN_STEPS =200
SEED = 0

with open(GENOME_PATH, "r") as f:
    genome = json.load(f)

mean_reward = get_fitness_score(
    genome,
    timesteps=TRAIN_STEPS,
    seed=SEED,
    save_checkpoints=True,
    eval_during_train=False,
    log=False
)
