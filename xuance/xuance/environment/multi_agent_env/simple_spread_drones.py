"""
SimpleSpread for Drones — 纯 2D 粒子环境（无 PyBullet 依赖）

3 个 agent（代表无人机 2D 投影）需无碰撞地覆盖 3 个 landmark。
动作为 2D 位移，直接对应部署时 Navigate 模型的飞行目标。

参照 MPE simple_spread 设计：
  - 合作任务，所有 agent 共享奖励
  - 奖励 = -Σ min_agent_dist(landmark) - collision_penalty
"""
import io
import numpy as np
from gymnasium.spaces import Box
from xuance.environment import RawMultiAgentEnv


class SimpleSpreadDronesEnv(RawMultiAgentEnv):
    """
    Pure 2D cooperative spread environment for drone high-level planning.

    Each agent outputs a 2D displacement (dx, dy) ∈ [-1, 1]²,
    scaled by ``max_step`` to update its position.

    Parameters
    ----------
    config : Namespace
        Must contain ``env_id``; optionally ``num_agents``, ``num_landmarks``,
        ``world_size``, ``max_step``, ``collision_radius``, ``collision_penalty``,
        ``max_episode_steps``, ``render_mode``, ``env_seed``.
    """

    def __init__(self, config):
        super().__init__()

        self.num_agents = getattr(config, "num_agents", 3)
        self.num_landmarks = getattr(config, "num_landmarks", 3)
        self.world_size = getattr(config, "world_size", 1.0)
        self.max_step = getattr(config, "max_step", 0.15)
        self.collision_radius = getattr(config, "collision_radius", 0.15)
        self.collision_penalty = getattr(config, "collision_penalty", 1.0)
        self.max_episode_steps = getattr(config, "max_episode_steps", 25)
        self.render_mode = getattr(config, "render_mode", None)

        seed = getattr(config, "env_seed", None)
        self._rng = np.random.default_rng(seed)

        # Agent names
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # --- spaces ---
        # Observation per agent: own_pos(2) + own_vel(2)
        #   + rel_landmark(2*num_landmarks) + rel_other_agents(2*(num_agents-1))
        obs_dim = 2 + 2 + 2 * self.num_landmarks + 2 * (self.num_agents - 1)
        self.observation_space = {
            k: Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            for k in self.agents
        }

        # Action per agent: 2D displacement direction
        self.action_space = {
            k: Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            for k in self.agents
        }

        # Global state: all_pos(2*N) + all_vel(2*N) + all_landmarks(2*L)
        state_dim = 2 * self.num_agents + 2 * self.num_agents + 2 * self.num_landmarks
        self.state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)

        # Internal buffers
        self.agent_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.landmark_pos = np.zeros((self.num_landmarks, 2), dtype=np.float32)
        self._episode_step = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        ws = self.world_size
        self.agent_pos = self._rng.uniform(-ws, ws, (self.num_agents, 2)).astype(np.float32)
        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.landmark_pos = self._rng.uniform(-ws, ws, (self.num_landmarks, 2)).astype(np.float32)
        self._episode_step = 0
        return self._get_obs(), {}

    def step(self, actions):
        # Apply actions — position-based displacement
        for i, agent_key in enumerate(self.agents):
            act = np.asarray(actions[agent_key], dtype=np.float32)
            act = np.clip(act, -1.0, 1.0)
            displacement = act * self.max_step
            self.agent_vel[i] = displacement
            self.agent_pos[i] = np.clip(
                self.agent_pos[i] + displacement,
                -self.world_size, self.world_size
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

        fig, ax = plt.subplots(figsize=(4, 4), dpi=64)
        ws = self.world_size
        ax.set_xlim(-ws, ws)
        ax.set_ylim(-ws, ws)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        ax.tick_params(colors='gray')

        colors = ['#e94560', '#0f3460', '#16213e']
        for li, lpos in enumerate(self.landmark_pos):
            ax.plot(lpos[0], lpos[1], '*', color='#f5a623', markersize=14, zorder=2)

        for ai, apos in enumerate(self.agent_pos):
            c = colors[ai % len(colors)]
            ax.plot(apos[0], apos[1], 'o', color=c, markersize=10, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.8)
            nearest_lm = self.landmark_pos[
                np.argmin(np.linalg.norm(self.landmark_pos - apos, axis=1))
            ]
            ax.annotate('', xy=nearest_lm, xytext=apos,
                        arrowprops=dict(arrowstyle='->', color=c, lw=0.8, alpha=0.5))

        ax.set_title(f"step {self._episode_step}", color='white', fontsize=9)
        ax.set_xlabel('x', color='gray', fontsize=7)
        ax.set_ylabel('y', color='gray', fontsize=7)

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
        # 1) Coverage: for each landmark, distance to the nearest agent
        reward = 0.0
        for lm in self.landmark_pos:
            dists = np.linalg.norm(self.agent_pos - lm, axis=1)
            reward -= float(np.min(dists))

        # 2) Collision penalty: for each pair of agents
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist_ij = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist_ij < self.collision_radius:
                    reward -= self.collision_penalty
        return reward
