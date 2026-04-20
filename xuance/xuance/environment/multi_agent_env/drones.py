"""
gym-pybullet-drones
GitHub: https://github.com/utiasDSL/gym-pybullet-drones.git
Note: The version of Python should be >= 3.10.
"""
import numpy as np
from gymnasium.spaces import Box
import time
from operator import itemgetter
from xuance.environment import RawMultiAgentEnv
try:
    import pybullet as p
    from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
    from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary as MultiHoverAviary_Official
    from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
except ImportError:
    p = None
    DroneModel, Physics, ActionType, ObservationType = None, None, None, None
    MultiHoverAviary_Official = object
    BaseRLAviary = object


class MultiHoverAviary(MultiHoverAviary_Official):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X if hasattr(DroneModel, 'CF2X') else None,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB if hasattr(Physics, 'PYB') else None,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN if hasattr(ObservationType, 'KIN') else None,
                 act: ActionType = ActionType.RPM if hasattr(ActionType, 'RPM') else None,
                 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.TARGET_POS = np.array([[0, 0, 1],
                                    [0, 1, 1],
                                    [1, 0, 1],
                                    [0, 0, 2],
                                    [0, 1, 2],
                                    [1, 0, 2],
                                    [2, 0, 1],
                                    [0, 2, 1],
                                    [2, 0, 2],
                                    [0, 2, 2], ])
        self.NUM_TARGETS = self.NUM_DRONES
        self.space_range_x = [-10.0, 10.0]
        self.space_range_y = [-10.0, 10.0]
        self.space_range_z = [0.02, 10.0]
        self.pose_limit = np.pi - 0.2

        ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        target_pos = self.TARGET_POS[:self.NUM_TARGETS].reshape(self.NUM_TARGETS, 1, 3)
        current_pos = states[:, :3].reshape(1, self.NUM_DRONES, 3)
        relative_pos = target_pos - current_pos
        distance_matrix = np.linalg.norm(relative_pos, axis=-1)
        reward_team = -distance_matrix.min(axis=-1, keepdims=True).sum()
        rewards = np.ones([self.NUM_DRONES, 1]) * reward_team

        for i in range(self.NUM_DRONES):
            x, y, z = states[i][0], states[i][1], states[i][2]
            if (max(abs(states[i][7]), abs(states[i][8])) > self.pose_limit) and (
                    z < self.space_range_z[0] + 0.05):  # the drone fulls down
                rewards[i] -= 10
            for j in range(self.NUM_DRONES):  # penalize collision with each other
                if i == j: continue
                distance_ij = np.linalg.norm(states[i, :3] - states[j, :3])
                if distance_ij < 0.1:
                    rewards[i] -= 10

        return rewards

        ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            x, y, z = states[i][0], states[i][1], states[i][2]
            if (max(abs(states[i][7]), abs(states[i][8])) > self.pose_limit) and (z < self.space_range_z[0] + 0.05):
                # The drone is too tilted
                return True

        return False

        ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            x, y, z = states[i][0], states[i][1], states[i][2]
            if (x < self.space_range_x[0]) or (x > self.space_range_x[1]) or (y < self.space_range_y[0]) or (
                    y > self.space_range_y[1]) or (z < self.space_range_z[0]) or (
                    z > self.space_range_z[1]):  # out of range
                return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False


class SimpleSpreadAviary3D(BaseRLAviary):
    """Multi-agent RL: cooperative spread to cover landmarks using PyBullet physics.

    Reward, reset, and termination follow MPE simple_spread style.
    Action/observation spaces follow NavigateAviary (VEL actions, KIN observations).
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X if hasattr(DroneModel, 'CF2X') else None,
                 num_drones: int = 3,
                 num_landmarks: int = 3,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB if hasattr(Physics, 'PYB') else None,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN if hasattr(ObservationType, 'KIN') else None,
                 act: ActionType = ActionType.VEL if hasattr(ActionType, 'VEL') else None,
                 world_size: float = 2.0,
                 collision_radius: float = 0.15,
                 collision_penalty: float = 1.0,
                 flight_height: float = 1.0,
                 ):
        # Set before super().__init__() because it calls _addObstacles()
        self.EPISODE_LEN_SEC = 60  # wrapper's max_episode_steps=1600 controls truncation
        self.NUM_LANDMARKS = num_landmarks
        self.world_size = world_size
        self.collision_radius = collision_radius
        self.collision_penalty = collision_penalty
        self.flight_height = flight_height
        self.landmark_pos = np.zeros((num_landmarks, 3), dtype=np.float32)
        self._landmark_visual_ids = []

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def _addObstacles(self):
        """Randomize drone and landmark positions on each reset."""
        self._landmark_visual_ids = []
        ws = self.world_size
        h = self.flight_height
        # Randomize drone positions
        for i in range(self.NUM_DRONES):
            pos = [
                np.random.uniform(-ws, ws),
                np.random.uniform(-ws, ws),
                h,
            ]
            p.resetBasePositionAndOrientation(
                self.DRONE_IDS[i], pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
            )
            p.resetBaseVelocity(
                self.DRONE_IDS[i], [0, 0, 0], [0, 0, 0],
                physicsClientId=self.CLIENT,
            )
        # Randomize landmark positions
        for j in range(self.NUM_LANDMARKS):
            self.landmark_pos[j] = [
                np.random.uniform(-ws, ws),
                np.random.uniform(-ws, ws),
                h,
            ]
            if self.GUI:
                visual = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.05,
                    rgbaColor=[1, 0.65, 0, 0.8],
                    physicsClientId=self.CLIENT,
                )
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual,
                    basePosition=self.landmark_pos[j].tolist(),
                    physicsClientId=self.CLIENT,
                )
                self._landmark_visual_ids.append(body_id)

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.KIN:
            # Per drone: drone_state(9) + rel_landmarks(3*L) + rel_agents(3*(N-1)) + action_buffer
            core_dim = 9 + 3 * self.NUM_LANDMARKS + 3 * (self.NUM_DRONES - 1)
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                act_size = 4
            elif self.ACT_TYPE == ActionType.PID:
                act_size = 3
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                act_size = 1
            else:
                act_size = 4
            total_dim = core_dim + act_size * self.ACTION_BUFFER_SIZE
            lo = -np.inf
            hi = np.inf
            obs_lower = np.full((self.NUM_DRONES, total_dim), lo, dtype=np.float32)
            obs_upper = np.full((self.NUM_DRONES, total_dim), hi, dtype=np.float32)
            return Box(low=obs_lower, high=obs_upper, dtype=np.float32)
        print("[ERROR] SimpleSpreadAviary3D only supports KIN observations.")
        exit()

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.KIN:
            core_dim = 9 + 3 * self.NUM_LANDMARKS + 3 * (self.NUM_DRONES - 1)
            obs_core = np.zeros((self.NUM_DRONES, core_dim), dtype=np.float32)
            # Gather all drone positions for relative computation
            all_pos = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                # drone_state: rpy(3) + rpy_rate(3) + vel(3) = 9
                drone_state = np.hstack([state[7:10], state[10:13], state[13:16]])
                # relative positions to all landmarks
                rel_landmarks = (self.landmark_pos - state[0:3]).flatten()
                # relative positions to other agents
                other_idx = [j for j in range(self.NUM_DRONES) if j != i]
                rel_agents = (all_pos[other_idx] - state[0:3]).flatten()
                obs_core[i, :] = np.hstack([drone_state, rel_landmarks, rel_agents])
            # Append action buffer
            ret = obs_core.astype('float32')
            for k in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[k][j, :] for j in range(self.NUM_DRONES)])])
            return ret
        print("[ERROR] SimpleSpreadAviary3D only supports KIN observations.")
        exit()

    def _computeReward(self):
        """Cooperative spread reward (MPE simple_spread style)."""
        all_pos = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        # Coverage: for each landmark, distance to nearest agent
        reward = 0.0
        for lm in self.landmark_pos:
            dists = np.linalg.norm(all_pos - lm, axis=1)
            reward -= float(np.min(dists))
        # Collision penalty
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                dist_ij = np.linalg.norm(all_pos[i] - all_pos[j])
                if dist_ij < self.collision_radius:
                    reward -= self.collision_penalty
        # Shared reward, shape (NUM_DRONES, 1) for wrapper compatibility
        return np.full((self.NUM_DRONES, 1), reward, dtype=np.float32)

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"episode_step": self.step_counter}


REGISTRY = {
    "MultiHoverAviary": MultiHoverAviary,
    "SimpleSpreadAviary3D": SimpleSpreadAviary3D,
}


class Drones_MultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, config):
        super(Drones_MultiAgentEnv, self).__init__()
        # import scenarios of gym-pybullet-drones
        self.env_id = config.env_id
        self.gui = config.render  # Note: You cannot render multiple environments in parallel.
        self.sleep = config.sleep
        self.env_id = config.env_id

        kwargs_env = {'gui': self.gui}
        if self.env_id in ["MultiHoverAviary", "SimpleSpreadAviary3D"]:
            kwargs_env.update({'num_drones': config.num_drones,
                               'obs': ObservationType(config.obs_type),
                               'act': ActionType(config.act_type)})
        if self.env_id in ["SimpleSpreadAviary3D"]:
            kwargs_env.update({
                'num_landmarks': getattr(config, 'num_landmarks', config.num_drones),
                'world_size': getattr(config, 'world_size', 2.0),
                'collision_radius': getattr(config, 'collision_radius', 0.15),
                'collision_penalty': getattr(config, 'collision_penalty', 1.0),
                'flight_height': getattr(config, 'flight_height', 1.0),
            })
        self.env = REGISTRY[config.env_id](**kwargs_env)
        self.env.reset(seed=config.env_seed)
        self.num_agents = config.num_drones
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        if self.env_id == "SimpleSpreadAviary3D":
            n_lm = getattr(config, 'num_landmarks', config.num_drones)
            state_dim = 3 * self.num_agents + 3 * self.num_agents + 3 * n_lm
            self.state_space = Box(-np.inf, np.inf, shape=[state_dim, ])
        else:
            self.state_space = Box(-np.inf, np.inf, shape=[20, ])
        obs_shape_i = (self.env.observation_space.shape[-1],)
        act_shape_i = (self.env.action_space.shape[-1],)
        self.observation_space = {k: Box(-np.inf, np.inf, obs_shape_i) for k in self.agents}
        self.action_space = {k: Box(-np.inf, np.inf, act_shape_i, seed=config.env_seed) for k in self.agents}

        self.max_episode_steps = self.max_cycles = config.max_episode_steps
        self._episode_step = 0

    def space_reshape(self, gym_space):
        low = gym_space.low.reshape(-1)
        high = gym_space.high.reshape(-1)
        shape_obs = (gym_space.shape[-1],)
        return Box(low=low, high=high, shape=shape_obs, dtype=gym_space.dtype)

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        return np.zeros([2, 2, 2])

    def reset(self):
        obs, info = self.env.reset()
        info["episode_step"] = self._episode_step
        self._episode_step = 0
        obs_dict = {k: obs[i] for i, k in enumerate(self.agents)}
        return obs_dict, info

    def step(self, actions):
        actions_array = np.array(itemgetter(*self.agents)(actions))
        obs, reward, terminated, truncated, info = self.env.step(actions_array)
        obs_dict = {k: obs[i] for i, k in enumerate(self.agents)}
        terminated_dict = {k: terminated for i, k in enumerate(self.agents)}
        rewrds_dict = {k: reward[i, 0] for i, k in enumerate(self.agents)}

        self._episode_step += 1
        truncated = True if (self._episode_step >= self.max_episode_steps) else False
        info["episode_step"] = self._episode_step  # current episode step

        if self.gui:
            time.sleep(self.sleep)

        return obs_dict, rewrds_dict, terminated_dict, truncated, info

    def agent_mask(self):
        return {agent: True for agent in self.agents}  # 1 means available

    def state(self):
        if self.env_id == "SimpleSpreadAviary3D":
            states = np.array([self.env._getDroneStateVector(i) for i in range(self.num_agents)])
            all_pos = states[:, 0:3].flatten()
            all_vel = states[:, 13:16].flatten()
            landmarks = self.env.landmark_pos.flatten()
            return np.concatenate([all_pos, all_vel, landmarks]).astype(np.float32)
        return self.state_space.sample()

    def avail_actions(self):
        return
