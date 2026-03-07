import itertools
import numpy as np

from creature_env import CreatureEnv
from stable_baselines3 import PPO
import pybullet as p

from genome import load_genome_from_json
from genome import genome_to_urdf



JSON_PATH = "best.creature.json"
URDF_PATH = "best_creature.urdf"

# -------- SEARCH MODE --------
# "grid"   -> classic grid search
# "jitter" -> random jitter around FIXED
SEARCH_MODE = "jitter"

N_JITTER_RUNS = 100      # used only if SEARCH_MODE == "jitter"
JITTER_FRAC = 0.2       # ±%
SEED = 0

TOTAL_TIMESTEPS = 1200
N_EVAL_EPISODES = 5

# -------- GRID (used only in grid mode) --------
GRID = {
    # "progress": [4, 16, 64],
    # "orientation": [0.1, 0.5, 1, 10],
    # "alive": [0.1, 0.5, 1, 10],
    # "speed": [0.1, 0.5, 2, 10],
}

# -------- BASE WEIGHTS --------
FIXED = {
    "alive": 2.2137,
    "progress": 11.4412,
    "speed": 0.1635,
    "orientation":  0.2956,
    "drift": 0.0856,
    "angular": 0.0770,
    "height": 0.9704,
    "energy": 0.0325,
    "smoothness": 0.1041,
}


def jitter_weights(weights, jitter_frac=0.2, rng=None):
    """Apply multiplicative ± jitter to weights."""
    if rng is None:
        rng = np.random.default_rng()

    return {
        k: float(v * (1.0 + rng.uniform(-jitter_frac, jitter_frac)))
        for k, v in weights.items()
    }


def evaluate_model(model, env, n_episodes=5):
    distances = []
    lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        start_pos, _ = p.getBasePositionAndOrientation(
            env.robot_id, physicsClientId=env.client
        )

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        end_pos, _ = p.getBasePositionAndOrientation(
            env.robot_id, physicsClientId=env.client
        )

        distances.append(max(0.0, end_pos[0] - start_pos[0])) #only forward motion
        lengths.append(steps)

    return np.mean(distances), np.mean(lengths)



genome = load_genome_from_json(JSON_PATH)
genome_to_urdf(genome, URDF_PATH)

rng = np.random.default_rng(SEED)


search_weights = []

if SEARCH_MODE == "grid":
    print("Using GRID search")

    keys = list(GRID.keys())
    values = list(GRID.values())

    for combo in itertools.product(*values):
        weights = FIXED.copy()
        weights.update(dict(zip(keys, combo)))
        search_weights.append(weights)

elif SEARCH_MODE == "jitter":
    print(f"Using JITTER search ({N_JITTER_RUNS} runs, ±{int(JITTER_FRAC*100)}%)")

    for _ in range(N_JITTER_RUNS):
        weights = jitter_weights(FIXED, JITTER_FRAC, rng)
        search_weights.append(weights)

else:
    raise ValueError(f"Unknown SEARCH_MODE: {SEARCH_MODE}")




results = []

for i, weights in enumerate(search_weights, 1):

    print(f"\nRun {i}/{len(search_weights)} — Testing weights:")
    for k, v in weights.items():
        print(f"  {k:12s}: {v:.4f}")

    env = CreatureEnv(
        urdf_path=URDF_PATH,
        render_mode=None,
        reward_weights=weights
    )

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    avg_dist, avg_len = evaluate_model(model, env, N_EVAL_EPISODES)

    results.append({
        "weights": weights,
        "distance": avg_dist,
        "length": avg_len
    })

    print(f"→ Avg distance: {avg_dist:.2f}, Avg length: {avg_len:.1f}")


results.sort(key=lambda x: x["distance"], reverse=True)

print("\n=== BEST RESULTS ===")
for r in results[:5]:
    print(f"Distance: {r['distance']:.2f}, Length: {r['length']:.1f}")
    for k, v in r["weights"].items():
        print(f"  {k:12s}: {v:.4f}")
    print()
