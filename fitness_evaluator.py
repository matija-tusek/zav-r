import json
import re
import time
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from creature_env import CreatureEnv
from genome import genome_to_urdf

# ── Paths & constants ──────────────────────────────────────────────────────────
TEMP_URDF_PATH    = "being.urdf"
TRAINING_TIMESTEPS = 10_000
CHECKPOINT_DIR    = "./checkpoints"
BEST_MODEL_DIR    = "./best_model"
EVAL_LOG_DIR      = "./eval_logs"
RESULTS_DIR       = "./results"          # All CSV + PNG files land here
N_EVAL_EPISODES   = 10


# fitness = α·norm_mean_reward + β·norm_forward_distance + γ·norm_upright_time
ALPHA = 0.50   # weight for normalised mean reward
BETA  = 0.35   # weight for normalised forward distance
GAMMA = 0.15   # weight for normalised upright time


REWARD_RANGE   = (-200.0, 2000.0)
DISTANCE_RANGE = (  0.0,   20.0)
UPRIGHT_RANGE  = (  0.0,    1.0)




def _ensure_results_dir():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def _csv_path(experiment_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{experiment_name}.csv")


CSV_FIELDNAMES = [
    "timestamp", "experiment", "run_id", "seed",
    "timesteps", "alpha", "beta", "gamma",
    "mean_reward", "std_reward",
    "mean_distance", "std_distance",
    "mean_upright", "std_upright",
    "fitness_score",
]


def _append_csv_row(row: dict, experiment_name: str = "experiments"):
    _ensure_results_dir()
    path = _csv_path(experiment_name)
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_experiment_results(experiment_name: str = "experiments"):
    """
    Read the CSV for *experiment_name* and (re-)generate summary plots.
    Call this after every run or batch of runs.
    """
    _ensure_results_dir()
    path = _csv_path(experiment_name)
    if not os.path.isfile(path):
        print(f"[plot] No CSV found at {path}, skipping.")
        return

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    run_ids      = [int(r["run_id"])       for r in rows]
    fitness      = [float(r["fitness_score"]) for r in rows]
    rewards      = [float(r["mean_reward"])   for r in rows]
    distances    = [float(r["mean_distance"]) for r in rows]
    uprights     = [float(r["mean_upright"])  for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Experiment: {experiment_name}", fontsize=14, fontweight="bold")

    def _plot(ax, y, label, color):
        ax.plot(run_ids, y, marker="o", color=color, linewidth=1.5)
        ax.set_xlabel("Run ID")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.4)

    _plot(axes[0, 0], fitness,   "Composite Fitness",   "#2196F3")
    _plot(axes[0, 1], rewards,   "Mean Reward",         "#4CAF50")
    _plot(axes[1, 0], distances, "Mean Forward Distance","#FF9800")
    _plot(axes[1, 1], uprights,  "Mean Upright Fraction","#9C27B0")

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, f"{experiment_name}_summary.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved → {out_png}")


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



def _evaluate_detailed(model, env_raw: CreatureEnv, n_episodes: int = N_EVAL_EPISODES):
    """
    Run *n_episodes* deterministic episodes and return per-episode arrays of:
        rewards, forward_distances, upright_fractions
    """
    rewards, distances, uprights = [], [], []

    for _ in range(n_episodes):
        obs, _ = env_raw.reset()
        done = False
        ep_reward = 0.0
        total_steps = 0
        upright_steps = 0

        start_pos, _ = p.getBasePositionAndOrientation(
            env_raw.robot_id, physicsClientId=env_raw.client
        )

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_raw.step(action)
            ep_reward += reward
            total_steps += 1

            # Upright = not terminated by fall (approx: check orientation)
            _, orn = p.getBasePositionAndOrientation(
                env_raw.robot_id, physicsClientId=env_raw.client
            )
            rpy = p.getEulerFromQuaternion(orn)
            if abs(rpy[0]) < 0.5 and abs(rpy[1]) < 0.5:
                upright_steps += 1

            done = terminated or truncated

        end_pos, _ = p.getBasePositionAndOrientation(
            env_raw.robot_id, physicsClientId=env_raw.client
        )

        rewards.append(ep_reward)
        distances.append(max(0.0, end_pos[0] - start_pos[0]))
        uprights.append(upright_steps / max(1, total_steps))

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



_RUN_COUNTER = 0


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
    run_id: int | None = None,
    alpha: float = ALPHA,
    beta: float  = BETA,
    gamma: float = GAMMA,
) -> float:

    global _RUN_COUNTER
    _RUN_COUNTER += 1
    _run_id = run_id if run_id is not None else _RUN_COUNTER


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
        # We evaluate with seeds [0, 1, 2] for stability, then aggregate
        eval_seeds = [0, 1, 2]
        all_rewards, all_distances, all_uprights = [], [], []

        # Unwrap Monitor to get the raw CreatureEnv for detailed eval
        raw_env = env.env  # Monitor wraps CreatureEnv

        for eval_seed in eval_seeds:
            raw_env.reset(seed=eval_seed)
            ep_rewards, ep_distances, ep_uprights = _evaluate_detailed(
                model, raw_env, n_episodes=N_EVAL_EPISODES
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
            f"dist={mean_distance:.2f}±{std_distance:.2f}  "
            f"upright={mean_upright:.2f}±{std_upright:.2f}  "
            f"→ FITNESS={fitness:.4f}"
        )

        # ── Log to CSV ────────────────────────────────────────────────────────
        row = {
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "experiment":    experiment_name,
            "run_id":        _run_id,
            "seed":          seed if seed is not None else "None",
            "timesteps":     timesteps,
            "alpha":         alpha,
            "beta":          beta,
            "gamma":         gamma,
            "mean_reward":   round(mean_reward,   4),
            "std_reward":    round(std_reward,    4),
            "mean_distance": round(mean_distance, 4),
            "std_distance":  round(std_distance,  4),
            "mean_upright":  round(mean_upright,  4),
            "std_upright":   round(std_upright,   4),
            "fitness_score": round(fitness,       6),
        }
        _append_csv_row(row, experiment_name=experiment_name)

        # Regenerate plots after each logged run
        plot_experiment_results(experiment_name=experiment_name)

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