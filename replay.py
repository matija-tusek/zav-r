import argparse
import re
import os
import time
import pybullet
from stable_baselines3 import SAC
from creature_env import CreatureEnv
#from fitness_evaluator import replay_checkpoints
from genome import load_genome_from_json, genome_to_urdf

CHECKPOINT_DIR = "./checkpoints"
TEMP_URDF_PATH = "being.urdf"

def replay_checkpoints(json_genome, checkpoint_dir: str = CHECKPOINT_DIR):
    # Generiranje URDF-a od genoma
    genome_to_urdf(json_genome, TEMP_URDF_PATH)

    # Kreiranje okruženja s GUI-jem
    env = CreatureEnv(urdf_path=TEMP_URDF_PATH, render_mode="human")

    # Postavljanje kamere tako da robot bude vidljiv
    pybullet.resetDebugVisualizerCamera(
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