"""
gym-pybullet-drones
GitHub: https://github.com/utiasDSL/gym-pybullet-drones.git
Note: The version of Python should be >= 3.10.
"""
import time
import numpy as np
from gymnasium.spaces import Box
from xuance.environment import RawEnvironment
try:
    import pybullet as p
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary as HoverAviary_Official
    from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
except ImportError:
    p = None
    HoverAviary_Official = None
    BaseRLAviary = object

class HoverAviary(HoverAviary_Official):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
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
        self.EPISODE_LEN_SEC = 8  # ?
        super().__init__(drone_model=drone_model,
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
        self.TARGET_POS = np.array([0, 0, 1])
        self.space_range = [2.0, 2.0]
        self.pose_limit = np.pi - 0.2
        self.height_limit = [0.05, 5.0]

    ################################################################################

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        reward = max(0, (1 - np.linalg.norm(self.TARGET_POS - state[0:3])) * 20)
        return reward

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > self.space_range[0]) or (abs(state[1]) > self.space_range[1]):  # Out of range
            return True
        if (state[2] > self.height_limit[1]) or (state[2] < self.height_limit[0]):  # Out of height
            return True
        if (abs(state[7]) > self.pose_limit or abs(state[8]) > self.pose_limit) and (
                state[2] < self.height_limit[0]):  # Truncate when the drone is too tilted
            return True
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < .0001:
            return True
        else:
            return False


class NavigateAviary(BaseRLAviary):
    """Single agent RL problem: fly from a random point A to a random point B."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 ):
        self.EPISODE_LEN_SEC = 20
        self.space_range_x = [-2.0, 2.0]
        self.space_range_y = [-2.0, 2.0]
        self.space_range_z = [0.3, 2.0]
        self.pose_limit = 0.6
        self.min_start_target_dist = 0.05
        self.max_start_target_dist = 1.0
        self.reach_threshold = 0.05
        self.reach_bonus = 100.0

        self.TARGET_POS = np.array([0.0, 0.0, 1.0])
        self._reached_target = False
        self._prev_dist = 0.0
        self.target_visual_id = -1
        self.target_line_id = -1

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        self.SPEED_LIMIT = 0.3

    def _randomizeStartAndTarget(self):
        """Sample random start, then target within a nearby sphere (0.05~1.0m)."""
        start = np.array([
            np.random.uniform(self.space_range_x[0] + 0.5, self.space_range_x[1] - 0.5),
            np.random.uniform(self.space_range_y[0] + 0.5, self.space_range_y[1] - 0.5),
            np.random.uniform(self.space_range_z[0] + 0.1, self.space_range_z[1] - 0.3),
        ])
        while True:
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            radius = np.random.uniform(self.min_start_target_dist, self.max_start_target_dist)
            target = start + direction * radius
            target[0] = np.clip(target[0], self.space_range_x[0] + 0.3, self.space_range_x[1] - 0.3)
            target[1] = np.clip(target[1], self.space_range_y[0] + 0.3, self.space_range_y[1] - 0.3)
            target[2] = np.clip(target[2], self.space_range_z[0], self.space_range_z[1])
            if np.linalg.norm(start - target) >= self.min_start_target_dist:
                break
        self.TARGET_POS = target
        self._reached_target = False
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            start.tolist(),
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            [0, 0, 0], [0, 0, 0],
            physicsClientId=self.CLIENT,
        )
        self._prev_dist = np.linalg.norm(start - target)

    def _addObstacles(self):
        super()._addObstacles()
        self._randomizeStartAndTarget()
        if self.GUI:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.05,
                rgbaColor=[1, 0, 0, 0.8],
                physicsClientId=self.CLIENT,
            )
            self.target_visual_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=self.TARGET_POS.tolist(),
                physicsClientId=self.CLIENT,
            )

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.KIN:
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo]*12 for _ in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi]*12 for _ in range(self.NUM_DRONES)])
            for _ in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    act_size = 4
                elif self.ACT_TYPE == ActionType.PID:
                    act_size = 3
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    act_size = 1
                else:
                    act_size = 4
                obs_lower_bound = np.hstack([obs_lower_bound, np.full((self.NUM_DRONES, act_size), -1)])
                obs_upper_bound = np.hstack([obs_upper_bound, np.full((self.NUM_DRONES, act_size), +1)])
            return Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        print("[ERROR] NavigateAviary only supports KIN observations.")
        exit()

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.KIN:
            obs_12 = np.zeros((self.NUM_DRONES, 12))
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                drone_state = np.hstack([state[7:10], state[10:13], state[13:16]])
                rel_pos = self.TARGET_POS - state[0:3]
                obs_12[i, :] = np.hstack([drone_state, rel_pos])
            ret = obs_12.astype('float32')
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret,
                                 np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            if self.GUI and self.target_visual_id >= 0:
                drone_pos = self._getDroneStateVector(0)[0:3]
                self.target_line_id = p.addUserDebugLine(
                    lineFromXYZ=drone_pos.tolist(),
                    lineToXYZ=self.TARGET_POS.tolist(),
                    lineColorRGB=[0, 1, 0],
                    lineWidth=1,
                    lifeTime=0,
                    replaceItemUniqueId=self.target_line_id,
                    physicsClientId=self.CLIENT,
                )
            return ret
        print("[ERROR] NavigateAviary only supports KIN observations.")
        exit()

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        dist = np.linalg.norm(pos - self.TARGET_POS)

        r_approach = 10.0 * (self._prev_dist - dist)
        r_distance = -1.0 * dist
        reward = r_approach + r_distance
        self._prev_dist = dist

        if dist < self.reach_threshold and not self._reached_target:
            self._reached_target = True
            reward += self.reach_bonus

        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < self.reach_threshold:
            return True
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        x, y, z = state[0], state[1], state[2]
        if (x < self.space_range_x[0] or x > self.space_range_x[1] or
                y < self.space_range_y[0] or y > self.space_range_y[1] or
                z < self.space_range_z[0] or z > self.space_range_z[1]):
            return True
        if abs(state[7]) > self.pose_limit or abs(state[8]) > self.pose_limit:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(state[0:3] - self.TARGET_POS)
        return {
            "distance_to_target": dist,
            "target_pos": self.TARGET_POS.copy(),
            "reached": self._reached_target,
        }


REGISTRY = {
    "CtrlAviary": CtrlAviary,
    "HoverAviary": HoverAviary,
    "VelocityAviary": VelocityAviary,
    "NavigateAviary": NavigateAviary,
}


class Drone_Env(RawEnvironment):
    def __init__(self, config):
        super(Drone_Env, self).__init__()
        # import scenarios of gym-pybullet-drones
        self.env_id = config.env_id

        self.gui = config.render  # Note: You cannot render multiple environments in parallel.
        self.sleep = config.sleep
        self.env_id = config.env_id

        kwargs_env = {'gui': self.gui}
        if self.env_id in ["HoverAviary", "NavigateAviary"]:
            kwargs_env.update({'obs': ObservationType(config.obs_type),
                               'act': ActionType(config.act_type)})
        if self.env_id not in ["HoverAviary", "NavigateAviary"]:
            kwargs_env.update({'num_drones': config.num_drones})
        self.env = REGISTRY[config.env_id](**kwargs_env)
        self.env.reset(seed=config.env_seed)

        self._episode_step = 0
        self.observation_space = self.space_reshape(self.env.observation_space)
        self.action_space = self.space_reshape(self.env.action_space)
        self.max_episode_steps = config.max_episode_steps

    def space_reshape(self, gym_space):
        low = gym_space.low.reshape(-1)
        high = gym_space.high.reshape(-1)
        shape_obs = (gym_space.shape[-1], )
        return Box(low=low, high=high, shape=shape_obs, dtype=gym_space.dtype)

    def close(self):
        self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self):
        obs, info = self.env.reset()
        self._episode_step = 0
        obs_return = obs.reshape(-1)
        return obs_return, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions.reshape([1, -1]))
        obs_return = obs.reshape(-1)
        self._episode_step += 1
        truncated = truncated or (self._episode_step >= self.max_episode_steps)
        if self.gui:
            time.sleep(self.sleep)
        return obs_return, reward, terminated, truncated, info



