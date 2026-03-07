from numbers import Number
import time
import numpy as np
import pybullet as p
import pybullet_data

MAX_FORCE = 50.0


class Simulation:
    def __init__(self, steps_per_action: int = 20):
        # How many physics substeps per RL action
        self.SIMULATION_STEPS_PER_ACTION = int(max(1, steps_per_action))

        self.client = None
        self.motor_ids = []
        self.joint_limits = None  # np array shape (n_motors, 2) [low, high]

        self.planeShapeId = -1
        self.planeId = -1  # <-- IMPORTANT: this will be the PLANE BODY id now

        self._foot_links_cache = {}  # robot_id -> list[int]

    def setup_pybullet(self, gui: Number):
        gui = bool(gui)
        print("Connecting to PyBullet, GUI:", gui)


        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=self.client)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Create plane BODY (store body id, not shape id)
        self.planeShapeId = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        self.planeId = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.planeShapeId,
            physicsClientId=self.client
        )

        # Optional but often helps walking: add friction on ground
        p.changeDynamics(self.planeId, -1, lateralFriction=1.0, physicsClientId=self.client)

        return self.client

    def setup_creature_in_pybullet(self, path, client):
        # Keep client consistent
        self.client = client

        creature = p.loadURDF(
            path,
            basePosition=[0, 0, 0.25],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.client
        )

        motor_ids = []
        limits = []

        for i in range(p.getNumJoints(creature, physicsClientId=self.client)):
            info = p.getJointInfo(creature, i, physicsClientId=self.client)
            joint_type = info[2]

            if joint_type == p.JOINT_REVOLUTE:
                motor_ids.append(i)

                low = float(info[8])
                high = float(info[9])

                # Some URDFs leave limits at (0,0). If that happens, give a safe default.
                if abs(high - low) < 1e-6:
                    low, high = -1.0, 1.0

                limits.append((low, high))

        self.motor_ids = motor_ids
        self.joint_limits = np.asarray(limits, dtype=np.float32)

        # Cache foot links once
        self._foot_links_cache[creature] = self.get_foot_links(creature)

        return creature

    def get_foot_links(self, robot_id):
        # Use cache if present
        if robot_id in self._foot_links_cache:
            return self._foot_links_cache[robot_id]

        foot_links = []
        n = p.getNumJoints(robot_id, physicsClientId=self.client)
        for i in range(n):
            info = p.getJointInfo(robot_id, i, physicsClientId=self.client)
            link_name = info[12].decode("utf-8")
            if ("foot" in link_name.lower()) or ("end" in link_name.lower()):
                foot_links.append(i)

        self._foot_links_cache[robot_id] = foot_links
        return foot_links

    def apply_normalized_action(self, robot_id, action):
        action = np.asarray(action, dtype=np.float32)

        # Map [-1, 1] -> [low, high] per joint (vectorized)
        lows = self.joint_limits[:, 0]
        highs = self.joint_limits[:, 1]
        alpha = (action + 1.0) * 0.5  # [-1,1] -> [0,1]
        targets = lows + alpha * (highs - lows)

        p.setJointMotorControlArray(
            robot_id,
            self.motor_ids,
            p.POSITION_CONTROL,
            targetPositions=targets.tolist(),
            forces=[MAX_FORCE] * len(self.motor_ids),
            physicsClientId=self.client
        )

    def step_simulation(self, realtime=False):
        for _ in range(self.SIMULATION_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self.client)
            if realtime:
                time.sleep(1.0 / 240.0)

    def get_current_state(self, robot_id, motor_ids):
        states = []

        # Base
        pos, orient = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client)
        lin_speed, ang_speed = p.getBaseVelocity(robot_id, physicsClientId=self.client)

        states.extend(pos)                              # 3
        states.extend(p.getEulerFromQuaternion(orient)) # 3
        states.extend(lin_speed)                        # 3
        states.extend(ang_speed)                        # 3
        # total base = 12

        # Joints
        joint_states = p.getJointStates(robot_id, motor_ids, physicsClientId=self.client)
        states.extend([s[0] for s in joint_states])  # positions
        states.extend([s[1] for s in joint_states])  # velocities

        # Foot contacts (FAST): one contact query then mark links
        foot_links = self.get_foot_links(robot_id)
        if foot_links and self.planeId != -1:
            cps = p.getContactPoints(
                bodyA=robot_id,
                bodyB=self.planeId,
                physicsClientId=self.client
            )
            links_in_contact = {pt[3] for pt in cps}  # linkIndexA
            for link in foot_links:
                states.append(1.0 if link in links_in_contact else 0.0)

        return np.asarray(states, dtype=np.float32)

    def cleanup_pybullet(self, client):
        try:
            if p.isConnected(client):
                p.disconnect(client)
        except Exception:
            pass