import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pygad

from genome import genome_from_genes, save_genome_to_json
from fitness_evaluator import get_fitness_score, ALPHA, BETA, GAMMA

NUM_LEGS    = 4
TRAIN_STEPS = 200
POP         = 20
GENS        = 30

# ── Gene count ────────────────────────────────────────────────────────────────
#  Body:  7 genes  (x, y, z, mass, ixx, iyy, izz)
#  Leg:  14 genes  (upper: radius, length, mass, lower_lim, upper_lim, effort, velocity, stiffness)
#                  (lower: radius, length, mass, lower_lim, upper_lim, stiffness)
#  RGB removed — fixed colours, not evolved
NUM_GENES = 7 + NUM_LEGS * 14

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
PARALLEL  = ["process", N_WORKERS]

EXPERIMENT_NAME = "ga_run"

# ── Gene space ────────────────────────────────────────────────────────────────
gene_space = [
    # Body (7)
    {'low': 1.5,   'high': 3.0},    # 0  body x
    {'low': 0.6,   'high': 1.2},    # 1  body y
    {'low': 0.4,   'high': 0.8},    # 2  body z
    {'low': 8.0,   'high': 15.0},   # 3  body mass
    {'low': 0.001, 'high': 0.01},   # 4  ixx
    {'low': 0.001, 'high': 0.01},   # 5  iyy
    {'low': 0.001, 'high': 0.01},   # 6  izz
]

for _ in range(NUM_LEGS):
    gene_space += [
        # Upper segment (8)
        {'low': 0.1,   'high': 0.2},    # idx+0  upper radius
        {'low': 0.6,   'high': 1.0},    # idx+1  upper length
        {'low': 0.5,   'high': 2.0},    # idx+2  upper mass
        {'low': -3.14, 'high': -1.57},  # idx+3  upper joint lower limit
        {'low': 1.57,  'high': 3.14},   # idx+4  upper joint upper limit
        {'low': 50.0,  'high': 150.0},  # idx+5  upper joint effort
        {'low': 1.0,   'high': 3.0},    # idx+6  upper joint velocity
        {'low': 0.1,   'high': 2.0},    # idx+7  upper stiffness
        # Lower segment (6)
        {'low': 0.08,  'high': 0.15},   # idx+8  lower radius
        {'low': 0.8,   'high': 1.3},    # idx+9  lower length
        {'low': 0.5,   'high': 1.5},    # idx+10 lower mass
        {'low': -3.14, 'high': -1.0},   # idx+11 lower joint lower limit
        {'low': 1.0,   'high': 3.14},   # idx+12 lower joint upper limit
        {'low': 0.1,   'high': 2.0},    # idx+13 lower stiffness
    ]

# ── Progress tracking ─────────────────────────────────────────────────────────
best_fitness_each_gen = []


def fitness_func(ga_instance, solution, solution_idx):
    from genome import genome_from_genes
    from fitness_evaluator import get_fitness_score

    # Unique run_id across all generations and parallel workers:
    #   gen 0, solution 0  -> run_id =  0
    #   gen 0, solution 5  -> run_id =  5
    #   gen 1, solution 0  -> run_id = 20  (POP=20)
    gen    = ga_instance.generations_completed  # 0-based during evaluation
    run_id = gen * POP + solution_idx

    genome = genome_from_genes(solution, NUM_LEGS)
    fit = get_fitness_score(
        genome,
        timesteps=TRAIN_STEPS,
        save_checkpoints=False,
        eval_during_train=False,
        seed=solution_idx % 3,
        experiment_name=EXPERIMENT_NAME,
        run_id=run_id,
    )
    return float(fit)


def on_generation(ga_instance):
    # on_generation runs in the MAIN process — safe to do I/O and plotting here
    from fitness_evaluator import plot_experiment_results

    fitness = np.array(ga_instance.last_generation_fitness)
    best_idx = int(np.argmax(fitness))
    gen = ga_instance.generations_completed

    print(
        f"\n{'='*50}\n"
        f"Generacija {gen}/{GENS}  |  "
        f"Best = {fitness[best_idx]:.4f}  |  "
        f"Mean = {fitness.mean():.4f}  |  "
        f"Std = {fitness.std():.4f}\n"
        f"{'='*50}"
    )
    best_fitness_each_gen.append(fitness[best_idx])

    # Crash protection — spremi najboljeg nakon svake generacije
    best_sol, _, _ = ga_instance.best_solution()
    best_genome = genome_from_genes(best_sol, NUM_LEGS)
    save_genome_to_json(best_genome, f"best_creature_gen{gen}.json")

    # Regenerate summary plot once per generation (main process = safe)
    plot_experiment_results(experiment_name=EXPERIMENT_NAME)


if __name__ == "__main__":

    print(f"CPU jezgri: {os.cpu_count()}  |  Workeri: {N_WORKERS}")
    print(f"NUM_GENES: {NUM_GENES}  (bio 90, sada {NUM_GENES} — uklonjen RGB)")
    print(f"Pop: {POP}  |  Gen: {GENS}  |  Train koraci: {TRAIN_STEPS}\n")

    random.seed(0)
    np.random.seed(0)

    ga = pygad.GA(
        num_generations=GENS,
        sol_per_pop=POP,
        num_parents_mating=int(POP / 3),
        num_genes=NUM_GENES,
        gene_space=gene_space,
        fitness_func=fitness_func,
        on_generation=on_generation,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=15,
        keep_elitism=2,
        parallel_processing=PARALLEL,
    )

    # Initialise JSON experiment log (main process, before GA starts)
    from fitness_evaluator import init_experiment_log, finalise_experiment_log
    ga_settings = {
        "population_size":    POP,
        "num_generations":    GENS,
        "num_legs":           NUM_LEGS,
        "train_steps":        TRAIN_STEPS,
        "num_genes":          NUM_GENES,
        "parent_selection":   "tournament",
        "k_tournament":       3,
        "crossover_type":     "single_point",
        "mutation_type":      "random",
        "mutation_percent":   15,
        "keep_elitism":       2,
        "parallel_workers":   N_WORKERS,
    }
    init_experiment_log(EXPERIMENT_NAME, ga_settings, ALPHA, BETA, GAMMA)

    ga.run()

    # Finalise — writes summary stats to JSON
    finalise_experiment_log(EXPERIMENT_NAME)

    best_sol, best_fit, _ = ga.best_solution()
    print(f"\nBEST FITNESS: {best_fit:.4f}")

    best_genome = genome_from_genes(best_sol, NUM_LEGS)
    save_genome_to_json(best_genome, "best_creature.json")

    if best_fitness_each_gen:
        arr         = np.array(best_fitness_each_gen, dtype=np.float32)
        best_so_far = np.maximum.accumulate(arr)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"GA Evolucija — {EXPERIMENT_NAME}", fontsize=13, fontweight="bold")

        axes[0].plot(arr,         label="Best generacije", color="#2196F3", linewidth=2)
        axes[0].plot(best_so_far, label="Best do sada",    color="#FF5722", linewidth=2, linestyle="--")
        axes[0].set_xlabel("Generacija")
        axes[0].set_ylabel("Fitness")
        axes[0].set_title("Napredak fitnessa")
        axes[0].legend()
        axes[0].grid(True, alpha=0.4)

        last_gen = np.array(ga.last_generation_fitness)
        axes[1].hist(last_gen, bins=10, color="#4CAF50", edgecolor="white")
        axes[1].set_xlabel("Fitness")
        axes[1].set_ylabel("Broj jedinki")
        axes[1].set_title("Distribucija fitnessa (zadnja generacija)")
        axes[1].grid(True, alpha=0.4)

        plt.tight_layout()
        plt.savefig("ga_fitness_progress.png", dpi=150)
        plt.show()