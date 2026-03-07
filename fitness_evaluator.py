import json
import re
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
import numpy as np
import os
import shutil
import pybullet as p
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from creature_env import CreatureEnv
from create_urdf_from_json import genome_to_urdf

TEMP_URDF_PATH = "being.urdf"
TRAINING_TIMESTEPS = 10000
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_DIR = "./best_model"
EVAL_LOG_DIR = "./eval_logs"

from stable_baselines3.common.callbacks import BaseCallback


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


def get_fitness_score(json_genome, timesteps: int = TRAINING_TIMESTEPS, *,
                      save_checkpoints: bool = False, checkpoint_dir: str = CHECKPOINT_DIR,
                      eval_during_train: bool = False, seed: int | None = None,
                      ) -> float:
    # Generiranje i Učitavanje Fizičkog Modela ---

    try:
        # Pozivanje funkcije koja pretvara JSON u URDF
        genome_to_urdf(json_genome, TEMP_URDF_PATH)
    except Exception as e:
        print(f"ERROR: Neuspješno generiranje URDF-a: {e}")
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

        # Opcionalni checkpointovi (sporije odvijanje)
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

        # Opcionalna evaluacija za trajanje treninga (sporije odvijanje)
        if eval_during_train:
            if os.path.exists(BEST_MODEL_DIR):
                shutil.rmtree(BEST_MODEL_DIR)
            os.makedirs(BEST_MODEL_DIR, exist_ok=True)

            if os.path.exists(EVAL_LOG_DIR):
                shutil.rmtree(EVAL_LOG_DIR)
            os.makedirs(EVAL_LOG_DIR, exist_ok=True)

            eval_env = Monitor(CreatureEnv(urdf_path=TEMP_URDF_PATH, render_mode=None))
            stop_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5,
                min_evals=3,
                verbose=0
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

        # SAC (prilagođeno bržem učenju, može se modificirati)
        policy_kwargs = dict(net_arch=[256, 256])
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            device="auto",
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            buffer_size=max(50000, int(timesteps) * 10),
            learning_starts=max(10, int(timesteps) // 20),
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            seed=seed,
        )

        # treniraj
        if callbacks:
            model.learn(total_timesteps=int(timesteps), callback=callbacks)
        else:
            model.learn(total_timesteps=int(timesteps))

        # Ako smo spremili najbolji model, koristi ga za evaluaciju
        best_model_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
        if eval_during_train and os.path.exists(best_model_path):
            model = SAC.load(best_model_path, env=env)

        # Pravila evaluairanja
        mean_reward, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=5,
            deterministic=True,
            return_episode_rewards=False,
        )


        return float(mean_reward)

    except Exception as e:
        print(f"ERROR: Greška u RL treningu/evaluaciji: {e}")
        return 0.0

    finally:
        # Zatvori okruzenja
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

        # Čišćenje privremenog URDF-a
        try:
            if os.path.exists(TEMP_URDF_PATH):
                os.remove(TEMP_URDF_PATH)
        except Exception:
            pass


def evaluate_genome(genome_json) -> float:
    return get_fitness_score(genome_json)


def replay_checkpoints(json_genome, checkpoint_dir: str = CHECKPOINT_DIR):
    # Generiranje URDF-a od genoma
    genome_to_urdf(json_genome, TEMP_URDF_PATH)

    # Kreiranje okruženja s GUI-jem
    env = CreatureEnv(urdf_path=TEMP_URDF_PATH, render_mode="human")

    # Postavljanje kamere tako da robot bude vidljiv
    p.resetDebugVisualizerCamera(
        cameraDistance=4.5,  # udaljenost od cilja
        cameraYaw=30,  # horizontalna rotacija
        cameraPitch=-30,  # vertikalna rotacija
        cameraTargetPosition=[0, 0, 0.25],  # centar robota
        physicsClientId=env.client,  # ID PyBullet klijenta (povezan s ovom simulacijom)
    )

    def extract_steps(name):
        numbers = re.findall(r"\d+", name)
        return int(numbers[-1]) if numbers else 0
    # Učitavanje svih checkpointova
    checkpoints = sorted(
        (f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")),
        key=extract_steps
    )



    for ckpt in checkpoints:
        print(f"\n▶ Replay: {ckpt}")
        model = SAC.load(os.path.join(checkpoint_dir, ckpt), env=env)
        obs, info = env.reset()
        # povećajte broj koraka ako želite duže gledati
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # Pauza da se vidi simulacija (PyBullet radi u 240 Hz)
            time.sleep(1. / 120.)

            if terminated or truncated:
                break

    env.close()

    if os.path.exists(TEMP_URDF_PATH):
        os.remove(TEMP_URDF_PATH)