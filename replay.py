import argparse
import re
import os
import time
import pybullet
from stable_baselines3 import SAC
from creature_env import CreatureEnv
from genome import load_genome_from_json, genome_to_urdf

CHECKPOINT_DIR = "./checkpoints"
TEMP_URDF_PATH = "being.urdf"

# Keys (PyBullet key codes)
KEY_SKIP = ord(' ')   # Space — skip current checkpoint
KEY_QUIT = ord('q')   # Q     — quit all replays


'''
# Bez videa (kao i prije)
python replay.py --genome best_creature.json

# Sa videom u ./videos/
python replay.py --genome best_creature.json --save-video

# Sa videom u custom folderu
python replay.py --genome best_creature.json --save-video --video-dir results/exp1/videos

NAPOMENA: treba biti instaliran ffmpeg da snimanje videa
'''

def replay_checkpoints(json_genome, checkpoint_dir: str = CHECKPOINT_DIR,
                       save_video: bool = False, video_dir: str = "./videos"):
    genome_to_urdf(json_genome, TEMP_URDF_PATH)

    env = CreatureEnv(urdf_path=TEMP_URDF_PATH, render_mode="human")
    client = env.client

    pybullet.resetDebugVisualizerCamera(
        cameraDistance=4.5,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.25],
        physicsClientId=client,
    )

    if save_video:
        os.makedirs(video_dir, exist_ok=True)

    def extract_steps(name):
        numbers = re.findall(r"\d+", name)
        return int(numbers[-1]) if numbers else 0

    checkpoints = sorted(
        (f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")),
        key=extract_steps
    )

    print("\nControls (click the PyBullet window first to give it focus):")
    print("  Space — skip to next checkpoint")
    print("  Q     — quit replay\n")

    if save_video:
        print(f"  Video recording enabled → {video_dir}/\n")

    quit_replay = False

    for ckpt in checkpoints:
        if quit_replay:
            break

        print(f"▶ Replay: {ckpt}")

        # Start video recording for this checkpoint
        if save_video:
            ckpt_stem  = os.path.splitext(ckpt)[0]
            video_path = os.path.join(video_dir, f"{ckpt_stem}.mp4")
            pybullet.startStateLogging(
                pybullet.STATE_LOGGING_VIDEO_MP4,
                video_path,
                physicsClientId=client,
            )
            print(f"  🎥 Recording → {video_path}")

        model = SAC.load(os.path.join(checkpoint_dir, ckpt), env=env)
        obs, _ = env.reset()

        log_id = None
        skip = False
        for _ in range(1000):
            keys = pybullet.getKeyboardEvents(physicsClientId=client)

            if KEY_QUIT in keys and keys[KEY_QUIT] & pybullet.KEY_WAS_TRIGGERED:
                quit_replay = True
                break
            if KEY_SKIP in keys and keys[KEY_SKIP] & pybullet.KEY_WAS_TRIGGERED:
                skip = True
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            time.sleep(1. / 120.)

            if terminated or truncated:
                break

        # Stop recording for this checkpoint
        if save_video:
            pybullet.stopStateLogging(log_id if log_id else 0, physicsClientId=client)

        if quit_replay:
            print("\nQuit.")
        elif skip:
            print(f"  ⏭ Skipped: {ckpt}")
        else:
            print(f"  ✓ Done: {ckpt}")

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
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save an MP4 video of each checkpoint replay."
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos",
        help="Directory to save videos (default: ./videos)."
    )
    args = parser.parse_args()

    genome = load_genome_from_json(args.genome)
    print(f"Replaying checkpoints for genome: {args.genome}")
    replay_checkpoints(genome, save_video=args.save_video, video_dir=args.video_dir)


if __name__ == "__main__":
    main()