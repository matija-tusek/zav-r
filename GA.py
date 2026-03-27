import os
import sys
import argparse
import random
import numpy as np
import pygad

from genome import genome_from_genes, save_genome_to_json
from fitness_evaluator import get_fitness_score, ALPHA, BETA, GAMMA, _json_path


# ── CLI arguments (used when launched from run_all.py) ───────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--experiment", type=str, default=None)
_parser.add_argument("--run",        type=str, default=None)
_parser.add_argument("--seed",       type=int, default=0)
_args, _ = _parser.parse_known_args()

# Prevent Windows from sleeping or throttling during GA run
import ctypes
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040
ctypes.windll.kernel32.SetThreadExecutionState(
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
)
print("[power] Windows sleep/throttle prevention active")

NUM_LEGS    = 4
TRAIN_STEPS = 300
POP         = 36
GENS        = 24

# ── Gene count ────────────────────────────────────────────────────────────────
#
#  BODY  (4 genes)
#   0  body_x           0.8 – 2.0 m
#   1  body_y           0.5 – 1.0 m
#   2  body_z           0.3 – 0.7 m
#   3  body_mass        3.0 – 10.0 kg
#      ixx/iyy/izz  →  auto-calculated from geometry (not a gene)
#
#  PER LEG  (11 genes)
#   +0  upper_radius       0.06 – 0.18 m
#   +1  total_leg_length   0.8  – 2.0  m
#   +2  upper_ratio        0.35 – 0.65  (upper = total*ratio, lower = total*(1-ratio))
#   +3  upper_mass         0.4  – 1.5  kg
#   +4  upper_jnt_lower   -1.2 – -0.3  rad  (hip, realistic range)
#   +5  upper_jnt_upper    0.3  –  1.2  rad
#   +6  upper_stiffness    0.5  –  4.0
#   +7  lower_radius       0.05 – 0.12 m
#   +8  lower_mass         0.3  –  1.2 kg
#   +9  lower_jnt_lower   -2.0 – -0.5  rad  (knee bends backwards only)
#   +10 foot_length        0.15 –  0.5  m
#
#  REMOVED vs previous version:
#    - ixx/iyy/izz (auto-calculated)
#    - upper effort & velocity (effort = mass*30, velocity fixed at 2.0)
#    - lower effort & velocity (same)
#    - lower stiffness (fixed at 1.5)
#    - separate upper/lower lengths replaced by total_length + ratio
#
from genome import BODY_GENES, LEG_GENES
NUM_GENES = BODY_GENES + NUM_LEGS * LEG_GENES   # 4 + 4*11 = 48

# EXPERIMENT_NAME and GA_SEED can be overridden via --experiment / --run / --seed CLI args
import pathlib

GA_SEED         = _args.seed
EXPERIMENT_NAME = _args.experiment if _args.experiment else f"exp_seed{GA_SEED}"
RUN_NAME        = _args.run        if _args.run        else "run_1"

# All output for this run goes into: results/<experiment>/<run>/
RUN_DIR = pathlib.Path("results") / EXPERIMENT_NAME / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Convenience: full path prefix for all output files of this run
# e.g. RUN_PREFIX = "results/eksperiment_1/run_1/run_1"
RUN_PREFIX = str(RUN_DIR / RUN_NAME)

# experiment_name passed to fitness_evaluator — points to the JSON inside RUN_DIR
FE_EXPERIMENT_NAME = str(pathlib.Path(EXPERIMENT_NAME) / RUN_NAME / RUN_NAME)

# ── Gene space ────────────────────────────────────────────────────────────────
gene_space = [
    # Body (4)
    {'low': 0.8,  'high': 2.0},   # 0  body_x
    {'low': 0.5,  'high': 1.0},   # 1  body_y
    {'low': 0.3,  'high': 0.7},   # 2  body_z
    {'low': 3.0,  'high': 10.0},  # 3  body_mass
]

for _ in range(NUM_LEGS):
    gene_space += [
        # Upper segment
        {'low': 0.06,  'high': 0.18},   # +0  upper_radius
        {'low': 0.8,   'high': 2.0},    # +1  total_leg_length
        {'low': 0.35,  'high': 0.65},   # +2  upper_ratio
        {'low': 0.4,   'high': 1.5},    # +3  upper_mass
        {'low': -1.2,  'high': -0.3},   # +4  upper_jnt_lower  (hip)
        {'low': 0.3,   'high': 1.2},    # +5  upper_jnt_upper  (hip)
        {'low': 0.5,   'high': 4.0},    # +6  upper_stiffness
        # Lower segment
        {'low': 0.05,  'high': 0.12},   # +7  lower_radius
        {'low': 0.3,   'high': 1.2},    # +8  lower_mass
        {'low': -2.0,  'high': -0.5},   # +9  lower_jnt_lower  (knee, bends back)
        # Foot
        {'low': 0.15,  'high': 0.5},    # +10 foot_length
    ]

# ── Progress tracking ─────────────────────────────────────────────────────────
best_fitness_each_gen = []


def fitness_func(ga_instance, solution, solution_idx):
    from genome import genome_from_genes
    from fitness_evaluator import get_fitness_score

    gen    = ga_instance.generations_completed
    run_id = f"Gen{gen}Creature{solution_idx}"

    genome = genome_from_genes(solution, NUM_LEGS)
    fit = get_fitness_score(
        genome,
        timesteps=TRAIN_STEPS,
        save_checkpoints=False,
        eval_during_train=False,
        seed=GA_SEED,
        experiment_name=FE_EXPERIMENT_NAME,
        run_id=run_id,
    )
    return float(fit)


def on_generation(ga_instance):
    # on_generation runs in the MAIN process — safe to do I/O and plotting here

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
    save_genome_to_json(best_genome, RUN_PREFIX + "_best_creature.json")

    # Regenerate summary plot once per generation
    try:
        from plot_results import plot_summary
        plot_summary(_json_path(FE_EXPERIMENT_NAME))
    except Exception as e:
        print(f"[plot] WARNING: summary plot failed: {e}")


if __name__ == "__main__":

    print(f"[{EXPERIMENT_NAME}/{RUN_NAME}] starting  seed={GA_SEED}")
    print(f"NUM_GENES: {NUM_GENES}  (BODY={BODY_GENES} + {NUM_LEGS} noge x {LEG_GENES} = {NUM_GENES})")
    print(f"Pop: {POP}  |  Gen: {GENS}  |  Train koraci: {TRAIN_STEPS}\n")

    random.seed(GA_SEED)
    np.random.seed(GA_SEED)

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
        keep_elitism=2
    )

    # Initialise JSON experiment log (main process, before GA starts)
    from fitness_evaluator import init_experiment_log, finalise_experiment_log
    ga_settings = {
        "population_size":    POP,
        "num_generations":    GENS,
        "num_legs":           NUM_LEGS,
        "train_steps":        TRAIN_STEPS,
        "num_genes":          NUM_GENES,
        "body_genes":         BODY_GENES,
        "leg_genes_per_leg":  LEG_GENES,
        "parent_selection":   "tournament",
        "k_tournament":       3,
        "crossover_type":     "single_point",
        "mutation_type":      "random",
        "mutation_percent":   15,
        "keep_elitism":       2,
    }
    init_experiment_log(FE_EXPERIMENT_NAME, ga_settings, ALPHA, BETA, GAMMA)

    ga.run()

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    # Finalise — writes summary stats to JSON
    finalise_experiment_log(FE_EXPERIMENT_NAME)

    best_sol, best_fit, _ = ga.best_solution()
    print(f"\nBEST FITNESS: {best_fit:.4f}")

    best_genome = genome_from_genes(best_sol, NUM_LEGS)
    save_genome_to_json(best_genome, RUN_PREFIX + "_best_creature.json")

    from plot_results import plot_progress
    plot_progress(
        out_path              = RUN_PREFIX + "_progress.png",
        best_fitness_each_gen = best_fitness_each_gen,
        last_gen_fitness      = ga.last_generation_fitness.tolist(),
        title                 = f"GA Evolucija — {EXPERIMENT_NAME} / {RUN_NAME}",
    )