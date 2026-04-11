import json
import re
import time
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pybullet as p

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from creature_env import CreatureEnv
from genome import genome_to_urdf

# ── Paths & constants ──────────────────────────────────────────────────────────
# Each process writes its URDF to a shared temp folder with a unique filename
_TEMP_DIR = "temp_urdf"
os.makedirs(_TEMP_DIR, exist_ok=True)
TEMP_URDF_PATH = os.path.join(_TEMP_DIR, f"being_{os.getpid()}.urdf")
TRAINING_TIMESTEPS = 10_000
CHECKPOINT_DIR    = "./checkpoints"
BEST_MODEL_DIR    = "./best_model"
EVAL_LOG_DIR      = "./eval_logs"
RESULTS_DIR       = "./results"          # All CSV + PNG files land here
N_EVAL_EPISODES   = 10                  # Changed: was 5, now 10

# ── Composite fitness weights ──────────────────────────────────────────────────
# fitness = α·norm_mean_reward + β·norm_forward_distance + γ·norm_upright_time
ALPHA = 1/7   # weight for normalised mean reward
BETA  = 4/7 # weight for normalised forward distance
GAMMA = 2/7  # weight for normalised upright time

# Reference ranges used for min-max normalisation (tune as data accumulates)
# Distance is measured in body lengths (metres / body_x gene).
# Range 0-10 means 0 to 10x the creature's own body length forward.
REWARD_RANGE   = (-200.0, 2000.0)
DISTANCE_RANGE = (  0.0,   10.0)   # body lengths
UPRIGHT_RANGE  = (  0.0,    1.0)   # fraction of steps robot was upright


# ══════════════════════════════════════════════════════════════════════════════
#  Logging helpers  (JSON-based, replaces CSV)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_results_dir(experiment_name: str = None):
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    if experiment_name:
        # Support subfolders e.g. "eksperiment_1/run_1"
        subfolder = os.path.dirname(os.path.join(RESULTS_DIR, experiment_name))
        Path(subfolder).mkdir(parents=True, exist_ok=True)


def _json_path(experiment_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{experiment_name}.json")


def init_experiment_log(experiment_name: str, ga_settings: dict,
                        alpha: float, beta: float, gamma: float) -> None:
    """
    Create a fresh JSON log file for one GA run.
    Call this ONCE from the main process before GA starts.

    Structure written:
    {
        "experiment": {
            "name": ...,
            "started_at": ...,
            "ga_settings": { population, generations, num_legs, ... },
            "fitness_weights": { alpha, beta, gamma },
            "normalisation_ranges": { reward, distance, upright }
        },
        "creatures": [],      <- filled by append_creature_log()
        "summary": {}         <- filled by finalise_experiment_log()
    }
    """
    _ensure_results_dir(experiment_name)
    doc = {
        "experiment": {
            "name":       experiment_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ga_settings": ga_settings,
            "fitness_weights": {
                "alpha (reward weight)":   alpha,
                "beta  (distance weight)": beta,
                "gamma (upright weight)":  gamma,
            },
            "normalisation_ranges": {
                "reward":   {"min": REWARD_RANGE[0],   "max": REWARD_RANGE[1]},
                "distance": {"min": DISTANCE_RANGE[0], "max": DISTANCE_RANGE[1]},
                "upright":  {"min": UPRIGHT_RANGE[0],  "max": UPRIGHT_RANGE[1]},
            },
        },
        "creatures": [],
        "summary":   {},
    }
    with open(_json_path(experiment_name), "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[log] Experiment log created → {_json_path(experiment_name)}")


def append_creature_log(creature_row: dict, experiment_name: str) -> None:
    """
    Append one creature's result to the JSON log.
    Uses a filelock so parallel child processes don't corrupt the file.
    Install filelock once with: pip install filelock
    """
    from filelock import FileLock
    path      = _json_path(experiment_name)
    lock_path = path + ".lock"

    with FileLock(lock_path):
        with open(path, "r") as f:
            doc = json.load(f)
        doc["creatures"].append(creature_row)
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)


def finalise_experiment_log(experiment_name: str) -> None:
    """
    Calculate summary statistics over all creatures and write them
    into the "summary" section of the JSON log.
    Call this ONCE from the main process after GA finishes.
    """
    path = _json_path(experiment_name)
    if not os.path.isfile(path):
        print(f"[log] No JSON found at {path}, skipping finalise.")
        return

    with open(path, "r") as f:
        doc = json.load(f)

    creatures = doc.get("creatures", [])
    if not creatures:
        return

    # Sort by generation then creature index, parsed from "GenXCreatureY"
    def _sort_key(c):
        rid = c.get("run_id", "Gen0Creature0")
        try:
            parts = str(rid).replace("Gen", "").split("Creature")
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (0, 0)
    creatures.sort(key=_sort_key)
    doc["creatures"] = creatures

    def _stat(key):
        vals = [c[key] for c in creatures if key in c]
        return {"mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "min":  round(float(np.min(vals)),  4),
                "max":  round(float(np.max(vals)),  4)}

    # Best creature overall
    best = max(creatures, key=lambda c: c.get("fitness_score", -999))

    doc["summary"] = {
        "finished_at":      datetime.now().isoformat(timespec="seconds"),
        "total_creatures":  len(creatures),
        "fitness_score":    _stat("fitness_score"),
        "mean_reward":      _stat("mean_reward"),
        "mean_distance":    _stat("mean_distance"),
        "mean_upright":     _stat("mean_upright"),
        "best_creature": {
            "run_id":       best.get("run_id"),
            "generation":   best.get("generation"),
            "fitness_score":best.get("fitness_score"),
            "mean_reward":  best.get("mean_reward"),
            "mean_distance":best.get("mean_distance"),
            "mean_upright": best.get("mean_upright"),
        },
    }

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[log] Experiment finalised → {path}")


def plot_experiment_results(experiment_name: str = "experiments") -> None:
    """Delegates to plot_results.plot_summary — all graph logic lives there."""
    from plot_results import plot_summary
    path = _json_path(experiment_name)
    plot_summary(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Callback – records per-episode reward for internal use
# ══════════════════════════════════════════════════════════════════════════════

class EpisodeRewardRecorder(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  Detailed evaluation (reward + distance + upright fraction)
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_detailed(model, env_raw: CreatureEnv, n_episodes: int = N_EVAL_EPISODES,
                       body_length: float = 1.0):
    """
    Run *n_episodes* deterministic episodes and return per-episode arrays of:
        rewards, forward_distances (in body lengths), upright_fractions

    body_length — creature torso x-size in metres (body_x from genome).
                  Raw metres / body_length = body lengths travelled.
    """
    rewards, distances, uprights = [], [], []

    max_steps = getattr(env_raw, "max_episode_steps", 500)

    for _ in range(n_episodes):
        obs, _ = env_raw.reset()
        done = False
        ep_reward = 0.0
        total_steps = 0

        start_pos, _ = p.getBasePositionAndOrientation(
            env_raw.robot_id, physicsClientId=env_raw.client
        )

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_raw.step(action)
            ep_reward += reward
            total_steps += 1
            done = terminated or truncated

        end_pos, _ = p.getBasePositionAndOrientation(
            env_raw.robot_id, physicsClientId=env_raw.client
        )

        dist_metres = max(0.0, end_pos[0] - start_pos[0])
        rewards.append(ep_reward)
        distances.append(dist_metres / max(body_length, 0.01))
        # Upright fraction = how long the robot survived relative to max episode length.
        uprights.append(total_steps / max_steps)

    return np.array(rewards), np.array(distances), np.array(uprights)


def _normalise(value, lo, hi):
    """Min-max normalise; clamps to [0, 1]."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _composite_fitness(mean_reward, mean_distance, mean_upright,
                       alpha=ALPHA, beta=BETA, gamma=GAMMA):
    nr = _normalise(mean_reward,   *REWARD_RANGE)
    nd = _normalise(mean_distance, *DISTANCE_RANGE)
    nu = _normalise(mean_upright,  *UPRIGHT_RANGE)
    return alpha * nr + beta * nd + gamma * nu


# ══════════════════════════════════════════════════════════════════════════════
#  Main public API
# ══════════════════════════════════════════════════════════════════════════════

def get_fitness_score(
    json_genome,
    timesteps: int = TRAINING_TIMESTEPS,
    *,
    save_checkpoints: bool = False,
    checkpoint_dir: str = CHECKPOINT_DIR,
    eval_during_train: bool = False,
    seed: int | None = None,
    # --- Logging ---
    experiment_name: str = "experiments",
    run_id: str | None = None,   # e.g. "Gen0Creature3" passed from GA.py
    alpha: float = ALPHA,
    beta: float  = BETA,
    gamma: float = GAMMA,
    log: bool = True,            # set False to skip JSON logging (e.g. learnBest.py)
) -> float:
    """
    Train SAC on *json_genome*, evaluate for N_EVAL_EPISODES (=10) episodes
    using multiple seeds, and return a composite fitness score.

    The composite fitness is:
        fitness = alpha · norm_reward + beta · norm_distance + gamma · norm_upright

    All measurements are appended to  results/<experiment_name>.csv
    and summary plots are regenerated after each call.

    NOTE: run_id must be passed explicitly from the caller (e.g. GA.py) because
    each parallel child process has its own memory space and cannot share a
    global counter reliably.
    """
    # Fallback for direct calls outside GA
    if run_id is None:
        _run_id = f"run_{datetime.now().strftime('%H%M%S')}"
    else:
        _run_id = run_id

    # ── Generate URDF ──────────────────────────────────────────────────────────
    try:
        genome_to_urdf(json_genome, TEMP_URDF_PATH)
    except Exception as e:
        print(f"ERROR: URDF generation failed: {e}")
        return 0.0

    env = None
    eval_env = None

    try:
        env = Monitor(CreatureEnv(urdf_path=TEMP_URDF_PATH, render_mode=None))
        if seed is not None:
            env.reset(seed=seed)

        # ── Callbacks ─────────────────────────────────────────────────────────
        callbacks = []
        reward_recorder = EpisodeRewardRecorder()
        callbacks.append(reward_recorder)

        if save_checkpoints:
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_cb = CheckpointCallback(
                save_freq=max(1, timesteps // 5),
                save_path=checkpoint_dir,
                name_prefix="sac_creature",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
            callbacks.append(checkpoint_cb)

        if eval_during_train:
            if os.path.exists(BEST_MODEL_DIR):
                shutil.rmtree(BEST_MODEL_DIR)
            os.makedirs(BEST_MODEL_DIR, exist_ok=True)
            if os.path.exists(EVAL_LOG_DIR):
                shutil.rmtree(EVAL_LOG_DIR)
            os.makedirs(EVAL_LOG_DIR, exist_ok=True)

            eval_env = Monitor(CreatureEnv(urdf_path=TEMP_URDF_PATH, render_mode=None))
            stop_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5, min_evals=3, verbose=0
            )
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=BEST_MODEL_DIR,
                log_path=EVAL_LOG_DIR,
                eval_freq=max(1, timesteps // 10),
                n_eval_episodes=3,
                deterministic=True,
                render=False,
                callback_after_eval=stop_cb,
            )
            callbacks.append(eval_cb)

        # ── Train ─────────────────────────────────────────────────────────────
        policy_kwargs = dict(net_arch=[256, 256])
        model = SAC(
            "MlpPolicy", env,
            verbose=0, device="auto",
            learning_rate=3e-4, gamma=0.99, tau=0.005,
            batch_size=256,
            buffer_size=max(50_000, int(timesteps) * 10),
            learning_starts=max(10, int(timesteps) // 20),
            train_freq=1, gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            seed=seed,
        )
        model.learn(total_timesteps=int(timesteps), callback=callbacks)

        # Load best model if available
        best_model_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
        if eval_during_train and os.path.exists(best_model_path):
            model = SAC.load(best_model_path, env=env)

        # ── Evaluate across multiple seeds ────────────────────────────────────
        # Extract body length from genome for distance normalisation (metres → body lengths)
        try:
            body_length = float(json_genome["base_body"]["geometry"]["size"]["x"])
        except (KeyError, TypeError):
            body_length = 1.0

        eval_seeds = [0]
        all_rewards, all_distances, all_uprights = [], [], []

        # Unwrap Monitor to get the raw CreatureEnv for detailed eval
        raw_env = env.env  # Monitor wraps CreatureEnv

        for eval_seed in eval_seeds:
            raw_env.reset(seed=eval_seed)
            ep_rewards, ep_distances, ep_uprights = _evaluate_detailed(
                model, raw_env, n_episodes=N_EVAL_EPISODES,
                body_length=body_length
            )
            all_rewards.extend(ep_rewards.tolist())
            all_distances.extend(ep_distances.tolist())
            all_uprights.extend(ep_uprights.tolist())

        # Aggregate (mean of last 10 episodes per seed → already N_EVAL_EPISODES each)
        mean_reward   = float(np.mean(all_rewards))
        std_reward    = float(np.std(all_rewards))
        mean_distance = float(np.mean(all_distances))
        std_distance  = float(np.std(all_distances))
        mean_upright  = float(np.mean(all_uprights))
        std_upright   = float(np.std(all_uprights))

        fitness = _composite_fitness(mean_reward, mean_distance, mean_upright,
                                     alpha=alpha, beta=beta, gamma=gamma)

        print(
            f"[Fitness run {_run_id}] "
            f"reward={mean_reward:.2f}±{std_reward:.2f}  "
            f"dist={mean_distance:.2f}±{std_distance:.2f}BL  "
            f"upright={mean_upright:.2f}±{std_upright:.2f}  "
            f"→ FITNESS={fitness:.4f}"
        )

        # ── Log creature to JSON (process-safe via filelock) ────────────────
        # Parse generation from run_id string "GenXCreatureY", fallback to 0
        if isinstance(_run_id, str) and _run_id.startswith("Gen"):
            try:
                generation = int(_run_id.split("Creature")[0].replace("Gen", ""))
            except (ValueError, IndexError):
                generation = 0
        else:
            generation = 0
        creature_row = {
            "run_id":        _run_id,
            "generation":    generation,
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "seed":          seed if seed is not None else "None",
            "timesteps":     timesteps,
            "body_length_m": round(body_length, 4),
            "mean_reward":   round(mean_reward,   4),
            "std_reward":    round(std_reward,    4),
            "mean_distance": round(mean_distance, 4),  # body lengths
            "std_distance":  round(std_distance,  4),  # body lengths
            "mean_upright":  round(mean_upright,  4),
            "std_upright":   round(std_upright,   4),
            "fitness_score": round(fitness,       6),
        }
        if log:
            append_creature_log(creature_row, experiment_name=experiment_name)
        # NOTE: plot is NOT regenerated here — call plot_experiment_results()
        # from the main process (e.g. on_generation in GA.py) to avoid
        # matplotlib crashes and race conditions in child processes.

        return fitness

    except Exception as e:
        print(f"ERROR: RL training/evaluation failed: {e}")
        return 0.0

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            if eval_env is not None:
                eval_env.close()
        except Exception:
            pass
        try:
            if os.path.exists(TEMP_URDF_PATH):
                os.remove(TEMP_URDF_PATH)
        except Exception:
            pass


# ── Convenience wrapper (unchanged API for GA.py etc.) ────────────────────────
def evaluate_genome(genome_json, experiment_name: str = "experiments") -> float:
    return get_fitness_score(genome_json, experiment_name=experiment_name)