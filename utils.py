from agilerl.wrappers.pettingzoo_wrappers import PettingZooAutoResetParallelWrapper
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
from agilerl.utils.multiprocessing_env import VecEnv
import numpy as np
import copy
from iterative_rps import rps_v0
from iterative_rps import n_agent_rps_v0
from lbforaging import lbf_v0
from lbforaging import lbf_simple_v0

class PettingZooVectorizationParallelWrapper(PettingZooAutoResetParallelWrapper):

    def env_select(self, env_name):
        if env_name == "rps-v0":
            return rps_v0
        if env_name == "n-agent-rps-v0":
            return n_agent_rps_v0
        elif env_name == "lbf-v0":
            return lbf_v0
        elif env_name == "lbf-simple-v0":
            return lbf_simple_v0
        raise Exception('Currently unsupported env!')

    def __init__(self, env: str, n_envs: int):
        super().__init__(env=self.env_select(env).parallel_env(render_mode=None))
        self.num_envs = n_envs
        self.env = DummyVecEnv([lambda: self.env_select(env).parallel_env(render_mode=None) for _ in range(n_envs)])
        return

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        for id, env in enumerate(self.env.remotes):
            if len(env.agents) <= 0:
                ob, info = env.reset()
                for key, value in obs.items():
                    value[id] = ob[key]
                    infos[key][id] = info[key]
        return obs, rewards, terminations, truncations, infos

class DummyVecEnv(VecEnv):
    """Vectorized environment class that collects samples synchronously

    Args:
        env_fns (list): list of gym environments to run in subprocesses

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """

    def __init__(self, env_fns):
        self.env = env_fns[0]()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes = [env_fn() for env_fn in env_fns]
        self.results = None
        VecEnv.__init__(
            self,
            len(env_fns),
            self.env.possible_agents,
        )

    def seed(self, value):
        for i_remote, remote in enumerate(self.remotes):
            remote.seed(value + i_remote)

    def step_async(self, actions):
        actions = [{p_id: a for p_id, a in zip (remote.possible_agents, act)} for remote, act in zip(self.remotes, actions)]
        self.results = [remote.step(action) for remote, action in zip(self.remotes, actions)]

    def step_wait(self):
        results = self.results
        obs, rews, dones, truncs, infos = zip(*results)

        ret_obs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_rews_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_dones_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_truncs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_infos_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        for env_idx, _ in enumerate(obs):
            for agent_idx, possible_agent in enumerate(self.env.possible_agents):
                ret_obs_dict[possible_agent].append(obs[env_idx][possible_agent])
                ret_rews_dict[possible_agent].append(rews[env_idx][possible_agent])
                ret_dones_dict[possible_agent].append(dones[env_idx][possible_agent])
                ret_truncs_dict[possible_agent].append(truncs[env_idx][possible_agent])
                ret_infos_dict[possible_agent].append(infos[env_idx][possible_agent])

        for agent_idx, possible_agent in enumerate(self.env.possible_agents):
            for op_dict in [
                ret_obs_dict,
                ret_rews_dict,
                ret_dones_dict,
                ret_truncs_dict,
                ret_infos_dict,
            ]:
                op_dict[possible_agent] = np.stack(op_dict[possible_agent])
        return (
            ret_obs_dict,
            ret_rews_dict,
            ret_dones_dict,
            ret_truncs_dict,
            ret_infos_dict,
        )

    def reset(self, seed=None, options=None):
        self.results = [remote.reset(seed) for remote in self.remotes]
        obs, infos = zip(*self.results)
        ret_obs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_infos_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }

        for env_idx, _ in enumerate(obs):
            for agent_idx, possible_agent in enumerate(self.env.possible_agents):
                ret_obs_dict[possible_agent].append(obs[env_idx][possible_agent])
                ret_infos_dict[possible_agent].append(infos[env_idx][possible_agent])
        for agent_idx, possible_agent in enumerate(self.env.possible_agents):
            for op_dict in [
                ret_obs_dict,
                ret_infos_dict,
            ]:
                op_dict[possible_agent] = np.stack(op_dict[possible_agent])
        return (ret_obs_dict, ret_infos_dict)

    def render(self):
        self.remotes[0].render(None)

    def close(self):
        for remote in self.remotes:
            remote.close()

    def sample_personas(self, is_train, is_val=True, path="./"):
        return self.env.sample_personas(is_train=is_train, is_val=is_val, path=path)