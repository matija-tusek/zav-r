import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from simulation import Simulation


class CreatureEnv(gym.Env):
    metadata = {"render_modes": "human", "render_fps": 30}

    def __init__(self, urdf_path: str, render_mode: str | None = None,
                 frame_skip: int = 1, max_episode_steps: int = 300, #smanjeno sa 1000
                 settle_steps: int = 50,
                 reward_weights=None):


        super().__init__()


        default_weights = {
            "alive": 2.4011,
            "progress": 15.0160,
            "speed": 0.1751,
            "orientation":  0.2801,
            "drift": 0.0882,
            "angular": 0.0790,
            "height": 1.3399,
            "energy": 0.0310,
            "smoothness": 0.101,
        }
        if reward_weights is None:
            reward_weights = default_weights
        self.w_alive = reward_weights["alive"]
        self.w_progress = reward_weights["progress"]
        self.w_speed = reward_weights["speed"]
        self.w_orientation = reward_weights["orientation"]
        self.w_drift = reward_weights["drift"]
        self.w_angular = reward_weights["angular"]
        self.w_height = reward_weights["height"]
        self.w_energy = reward_weights["energy"]
        self.w_smoothness = reward_weights["smoothness"]

        # Inicijalizacija PyBullet-a
        self.render_mode = render_mode
        self.frame_skip = int(max(1, frame_skip))
        self.max_episode_steps = int(max(1, max_episode_steps))
        self.settle_steps = int(max(0, settle_steps))
        self._elapsed_steps = 0
        self.sim = Simulation()
        self.client = self.sim.setup_pybullet(gui=(render_mode == "human"))  # GUI samo ako 'human'
        self.robot_id = self.sim.setup_creature_in_pybullet(urdf_path, self.client)

        # Cache-amo idjeve motora
        self.motor_ids = list(self.sim.motor_ids)
        self.n_motors = len(self.motor_ids)
        if self.n_motors <= 0:
            raise ValueError("CreatureEnv: No motors found (sim.motor_ids is empty).")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_motors,), dtype=np.float32)

        obs0 = np.asarray(self.sim.get_current_state(self.robot_id, self.motor_ids), dtype=np.float32).ravel()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32)

        # Praćenje stanja
        self.prev_base_pos = None
        self.prev_action = np.zeros((self.n_motors,), dtype=np.float32)
        self.last_action = np.zeros((self.n_motors,), dtype=np.float32)

        print(
            f"CreatureEnv Inicijalizacija: {self.n_motors} motora. OBS_dim={self.observation_space.shape[0]}, "

            f"frame_skip={self.frame_skip}"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        # Higher spawn position
        p.resetBasePositionAndOrientation(
            self.robot_id, [0, 0, 0.5],  # <-- Changed from 0.25 to 0.5 to prevent jumping (glitching)
            [0, 0, 0, 1],
            physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.robot_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.client
        )

        # Reset joint states
        for jid in self.motor_ids:
            p.resetJointState(
                self.robot_id, jid,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=self.client
            )

        # More settling steps, prevents jumping in the beginning
        for _ in range(200):  # <-- Changed from 50 to 200
            p.stepSimulation(physicsClientId=self.client)

        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        self.prev_base_pos = np.array(base_pos, dtype=np.float32)

        self.prev_action[:] = 0.0
        self.last_action[:] = 0.0

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self._elapsed_steps += 1

        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action
        self.sim.apply_normalized_action(self.robot_id, action)
        for _ in range(self.frame_skip):
            self.sim.step_simulation(realtime=(self.render_mode == "human"))

        obs = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = {}

        self.prev_action = self.last_action.copy()
        if self._elapsed_steps <= 30:
            return obs, 0.0, False, False, {} #stablizacija na pocetku
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _get_observation(self):
        obs = np.asarray(self.sim.get_current_state(self.robot_id, self.motor_ids), dtype=np.float32).ravel()
        return obs

    def _calculate_reward(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)

        base_pos = np.array(base_pos, dtype=np.float32)
        rpy = p.getEulerFromQuaternion(base_orn)

        delta_x = 0.0
        if self.prev_base_pos is not None:
            delta_x = float(base_pos[0] - self.prev_base_pos[0])
        self.prev_base_pos = base_pos

        vx, vy, z = float(lin_vel[0]), float(lin_vel[1]), float(base_pos[2])

        #neka grid skripta za isprobavanje
        #grid-search -> promjena jedne po jedne

        #tehnicka dokumentacija
        #dijagram komponenti -> sta koji dio koristi, struktura
        #koji se algoritmi koriste u pojedinim dijelovima
        #prikazati neke grafove i videe za demonstraciju

        target_height = 0.25


        reward = (
                self.w_alive
                + self.w_progress * max(0.0, delta_x)
                + self.w_speed * max(0.0, vx)
                - self.w_orientation * (abs(rpy[0]) + abs(rpy[1]))
                - self.w_drift * abs(vy)
                - self.w_angular * (abs(ang_vel[0]) + abs(ang_vel[1]) + abs(ang_vel[2]))
                - self.w_height * abs(z-target_height)
                - self.w_energy * float(np.mean(np.square(self.last_action)))
                 - self.w_smoothness * float(np.mean(np.abs(self.last_action - self.prev_action)))
        )
        return float(reward)

    def _is_terminated(self):
        base_pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        rpy = p.getEulerFromQuaternion(orn)

        # previše pada
        if (abs(rpy[0]) > 0.5 or abs(rpy[1]) > 0.5) :
            return True

        # prenisko
        if base_pos[2] < 0.1:
            return True

        return False

    def close(self):
        try:
            self.sim.cleanup_pybullet(self.client)
        except Exception:
            pass