"""
SimpleSpread for Drones — 3D 粒子环境（无 PyBullet 依赖）

3 个 agent（代表无人机）需无碰撞地覆盖 3 个 landmark，在三维空间中运动。
动作为 6D 力向量（MPE 官方风格），动力学完全复现 MPE integrate_state。

参照 MPE simple_spread 设计：
  - 合作任务，所有 agent 共享奖励
  - 奖励 = -Σ min_agent_dist(landmark) - collision_penalty
"""
import io
import numpy as np
from gymnasium.spaces import Box
from xuance.environment import RawMultiAgentEnv


class SimpleSpreadDrones3DEnv(RawMultiAgentEnv):
    """
    3D cooperative spread environment for drone benchmarking.

    Each agent outputs a 6D force vector [−x, +x, −y, +y, −z, +z] ∈ [0, 1]⁶,
    converted to a 3D net force via ``force = [right-left, up-down, fwd-back]``.
    Physics follows MPE's ``integrate_state``: position updated with old velocity,
    then velocity damped and force applied.

    Parameters
    ----------
    config : Namespace
        Must contain ``env_id``; optionally ``num_agents``, ``num_landmarks``,
        ``world_size``, ``collision_radius``, ``collision_penalty``,
        ``max_episode_steps``, ``render_mode``, ``env_seed``,
        ``dt``, ``damping``, ``mass``, ``max_speed``, ``flight_height``.
    """

    def __init__(self, config):
        super().__init__()

        self.num_agents = getattr(config, "num_agents", 3)
        self.num_landmarks = getattr(config, "num_landmarks", 3)
        self.world_size = getattr(config, "world_size", 1.0)
        self.collision_radius = getattr(config, "collision_radius", 0.15)
        self.collision_penalty = getattr(config, "collision_penalty", 1.0)
        self.max_episode_steps = getattr(config, "max_episode_steps", 1600)
        self.render_mode = getattr(config, "render_mode", None)
        self.flight_height = getattr(config, "flight_height", 1.0)

        # MPE dynamics parameters
        self.dt = getattr(config, "dt", 0.1)
        self.damping = getattr(config, "damping", 0.25)
        self.mass = getattr(config, "mass", 1.0)
        self.max_speed = getattr(config, "max_speed", None)

        seed = getattr(config, "env_seed", None)
        self._rng = np.random.default_rng(seed)

        # Agent names
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # --- spaces ---
        # Observation per agent: own_pos(3) + own_vel(3)
        #   + rel_landmark(3*num_landmarks) + rel_other_agents(3*(num_agents-1))
        obs_dim = 3 + 3 + 3 * self.num_landmarks + 3 * (self.num_agents - 1)
        self.observation_space = {
            k: Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            for k in self.agents
        }

        # Action per agent: 6D force [−x, +x, −y, +y, −z, +z] ∈ [0, 1]
        self.action_space = {
            k: Box(0.0, 1.0, shape=(6,), dtype=np.float32)
            for k in self.agents
        }

        # Global state: all_pos(3*N) + all_vel(3*N) + all_landmarks(3*L)
        state_dim = 3 * self.num_agents + 3 * self.num_agents + 3 * self.num_landmarks
        self.state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)

        # Internal buffers
        self.agent_pos = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.agent_vel = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.landmark_pos = np.zeros((self.num_landmarks, 3), dtype=np.float32)
        self._episode_step = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        ws = self.world_size
        h = self.flight_height
        # Drone positions: XY random, Z fixed (matching demo_hierarchical_deploy.py)
        xy = self._rng.uniform(-ws, ws, (self.num_agents, 2)).astype(np.float32)
        self.agent_pos = np.column_stack([xy, np.full(self.num_agents, h, dtype=np.float32)])
        self.agent_vel = np.zeros((self.num_agents, 3), dtype=np.float32)
        # Landmark positions: XY random, Z fixed
        lm_xy = self._rng.uniform(-ws, ws, (self.num_landmarks, 2)).astype(np.float32)
        self.landmark_pos = np.column_stack([lm_xy, np.full(self.num_landmarks, h, dtype=np.float32)])
        self._episode_step = 0
        return self._get_obs(), {}

    def step(self, actions):
        # Convert 6D force actions to 3D net force, then apply MPE dynamics
        for i, agent_key in enumerate(self.agents):
            act = np.asarray(actions[agent_key], dtype=np.float32)
            act = np.clip(act, 0.0, 1.0)
            # 6D → 3D net force: [right-left, up-down, forward-backward]
            force = np.array([
                act[1] - act[0],  # +x - (-x)
                act[3] - act[2],  # +y - (-y)
                act[5] - act[4],  # +z - (-z)
            ], dtype=np.float32)

            # MPE integrate_state order:
            # 1) Position update with old velocity
            self.agent_pos[i] += self.agent_vel[i] * self.dt
            # 2) Velocity damping
            self.agent_vel[i] *= (1 - self.damping)
            # 3) Apply force
            self.agent_vel[i] += (force / self.mass) * self.dt

            # 4) Speed clamping (if max_speed is set)
            if self.max_speed is not None:
                speed = np.linalg.norm(self.agent_vel[i])
                if speed > self.max_speed:
                    self.agent_vel[i] = (self.agent_vel[i] / speed) * self.max_speed

            # 5) Position clamping
            self.agent_pos[i] = np.clip(
                self.agent_pos[i], -self.world_size, self.world_size
            )

        # Reward (shared, cooperative)
        reward = self._compute_reward()
        reward_dict = {k: reward for k in self.agents}

        # Terminated: never (same as MPE simple_spread)
        terminated_dict = {k: False for k in self.agents}

        self._episode_step += 1
        truncated = self._episode_step >= self.max_episode_steps

        info = {"episode_step": self._episode_step}
        return self._get_obs(), reward_dict, terminated_dict, truncated, info

    def state(self):
        return np.concatenate([
            self.agent_pos.flatten(),
            self.agent_vel.flatten(),
            self.landmark_pos.flatten(),
        ]).astype(np.float32)

    def agent_mask(self):
        return {k: True for k in self.agents}

    def avail_actions(self):
        return None

    def render(self, *args, **kwargs):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(5, 5), dpi=64)
        ax = fig.add_subplot(111, projection='3d')
        ws = self.world_size
        ax.set_xlim(-ws, ws)
        ax.set_ylim(-ws, ws)
        ax.set_zlim(-ws, ws)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')

        colors = ['#e94560', '#0f3460', '#16213e']
        for lpos in self.landmark_pos:
            ax.scatter(lpos[0], lpos[1], lpos[2], marker='*', color='#f5a623', s=120, zorder=2)

        for ai, apos in enumerate(self.agent_pos):
            c = colors[ai % len(colors)]
            ax.scatter(apos[0], apos[1], apos[2], marker='o', color=c, s=80, zorder=3,
                       edgecolors='white', linewidths=0.8)

        ax.set_title(f"step {self._episode_step}", color='white', fontsize=9)
        ax.set_xlabel('x', color='gray', fontsize=7)
        ax.set_ylabel('y', color='gray', fontsize=7)
        ax.set_zlabel('z', color='gray', fontsize=7)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        from PIL import Image
        img = np.array(Image.open(buf).convert('RGB'))
        return img

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        obs_dict = {}
        for i, agent_key in enumerate(self.agents):
            own_pos = self.agent_pos[i]
            own_vel = self.agent_vel[i]
            # Relative positions to all landmarks
            rel_landmarks = (self.landmark_pos - own_pos).flatten()
            # Relative positions to other agents
            other_idx = [j for j in range(self.num_agents) if j != i]
            rel_agents = (self.agent_pos[other_idx] - own_pos).flatten()
            obs_dict[agent_key] = np.concatenate([
                own_pos, own_vel, rel_landmarks, rel_agents
            ]).astype(np.float32)
        return obs_dict

    def _compute_reward(self):
        # 1) Coverage: for each landmark, distance to the nearest agent (3D Euclidean)
        reward = 0.0
        for lm in self.landmark_pos:
            dists = np.linalg.norm(self.agent_pos - lm, axis=1)
            reward -= float(np.min(dists))

        # 2) Collision penalty: for each pair of agents (3D distance)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist_ij = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist_ij < self.collision_radius:
                    reward -= self.collision_penalty
        return reward
