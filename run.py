import random
import numpy as np
import matplotlib.pyplot as plt

import pygad

from GA import genome_from_genes, save_genome_to_json
from fitness_evaluator import get_fitness_score, replay_checkpoints


NUM_LEGS = 4
TRAIN_STEPS =200
POP = 20
GENS = 30

# Total number of genes: 10 for body + 20 per leg
NUM_GENES = 10 + NUM_LEGS * 20

# Gene space
gene_space = [
    # Base body
    {'low': 1.5, 'high': 3.0},   # x
    {'low': 0.6, 'high': 1.2},   # y
    {'low': 0.4, 'high': 0.8},   # z
    {'low': 8.0, 'high': 15.0},  # mass
    {'low': 0.001, 'high': 0.01},  # ixx
    {'low': 0.001, 'high': 0.01},  # iyy
    {'low': 0.001, 'high': 0.01},  # izz
    {'low': 0.0, 'high': 1.0},   # color R
    {'low': 0.0, 'high': 1.0},   # color G
    {'low': 0.0, 'high': 1.0},   # color B
]

for _ in range(NUM_LEGS):
    gene_space += [
        {'low': 0.1, 'high': 0.2},     # upper radius
        {'low': 0.6, 'high': 1.0},     # upper length
        {'low': 0.5, 'high': 2.0},     # upper mass
        {'low': -3.14, 'high': -1.57}, # joint lower
        {'low': 1.57, 'high': 3.14},   # joint upper
        {'low': 50.0, 'high': 150.0},  # joint effort
        {'low': 1.0, 'high': 3.0},     # joint velocity
        {'low': 0.1, 'high': 2.0},     # stiffness
        {'low': 0.0, 'high': 1.0},     # color R
        {'low': 0.0, 'high': 1.0},     # color G
        {'low': 0.0, 'high': 1.0},     # color B
        {'low': 0.08, 'high': 0.15},   # lower radius
        {'low': 0.8, 'high': 1.3},     # lower length
        {'low': 0.5, 'high': 1.5},     # lower mass
        {'low': -3.14, 'high': -1.0},  # joint lower
        {'low': 1.0, 'high': 3.14},    # joint upper
        {'low': 50.0, 'high': 150.0},  # joint effort
        {'low': 1.0, 'high': 3.0},     # joint velocity
        {'low': 0.1, 'high': 2.0},     # stiffness
        {'low': 0.3, 'high': 0.6},     # foot size x
    ]

# praćenje progressa od GA
best_fitness_each_gen = []

def fitness_func(ga_instance, solution, solution_idx):
    genome = genome_from_genes(solution, NUM_LEGS)
    fit = get_fitness_score(
        genome,
        timesteps=TRAIN_STEPS,
        save_checkpoints=False,
        eval_during_train=False,
        seed=0, #maybe change to solution_idx
    )

    return float(fit)

def on_generation(ga_instance):
    fitness = np.array(ga_instance.last_generation_fitness)
    best_idx = int(np.argmax(fitness))


    print(
        f"Gen {ga_instance.generations_completed}: "
        f"best fitness = {fitness[best_idx]:.3f}"
    )
    best_fitness_each_gen.append(fitness[best_idx])


random.seed(0)
np.random.seed(0)


ga = pygad.GA(
    num_generations=GENS,
    sol_per_pop=POP,
    num_parents_mating=int(POP/3),
    num_genes=NUM_GENES,
    gene_space=gene_space,
    fitness_func=fitness_func,
    on_generation=on_generation,
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="single_point", #uniform?
    mutation_type="random",
    mutation_percent_genes=15,
    keep_elitism=1
)


ga.run()


best_sol, best_fit, _ = ga.best_solution()
print("BEST FITNESS:", best_fit)

# Convert best solution genes to genome
best_genome = genome_from_genes(best_sol,NUM_LEGS)
save_genome_to_json(best_genome, "best_creature.json")

# graf fitnessa (uzima se samo najbolji)
if best_fitness_each_gen:
    arr = np.array(best_fitness_each_gen, dtype=np.float32)
    best_so_far = np.maximum.accumulate(arr)

    plt.figure()
    plt.plot(arr, label="Best of generation")
    plt.plot(best_so_far, label="Best so far", linestyle="--")
    plt.xlabel("Generacija")
    plt.ylabel("Fitness")
    plt.title("Reward Progress")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ga_fitness_progress.png", dpi=150)
    plt.show()


# Replay the best genome
#replay_checkpoints(best_genome)