from agilerl.wrappers.pettingzoo_wrappers import PettingZooAutoResetParallelWrapper
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
from agilerl.utils.multiprocessing_env import VecEnv
import numpy as np
import copy
from iterative_rps import rps_v0
from iterative_rps import n_agent_rps_v0
from lbforaging import lbf_v0
from lbforaging import lbf_simple_v0
from lbforaging import lbf_general_v0
from stag_hunt import state_based_n_agent_stag_hunt_v0
from stag_hunt import n_agent_stag_hunt_v0

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

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
        elif env_name == "lbf-general-v0":
            return lbf_general_v0
        elif env_name == "stag-hunt-simple-v0":
            return n_agent_stag_hunt_v0
        elif env_name == "stag-hunt-general-v0":
            return state_based_n_agent_stag_hunt_v0
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
    
def radar_factory(num_vars, frame='circle'):
	"""
	Create a radar chart with `num_vars` Axes.

	This function creates a RadarAxes projection and registers it.

	Parameters
	----------
	num_vars : int
		Number of variables for radar chart.
	frame : {'circle', 'polygon'}
		Shape of frame surrounding Axes.

	"""
	# calculate evenly-spaced axis angles
	theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

	class RadarTransform(PolarAxes.PolarTransform):

		def transform_path_non_affine(self, path):
			# Paths with non-unit interpolation steps correspond to gridlines,
			# in which case we force interpolation (to defeat PolarTransform's
			# autoconversion to circular arcs).
			if path._interpolation_steps > 1:
				path = path.interpolated(num_vars)
			return Path(self.transform(path.vertices), path.codes)

	class RadarAxes(PolarAxes):

		name = 'radar'
		PolarTransform = RadarTransform

		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			# rotate plot such that the first axis is at the top
			self.set_theta_zero_location('N')

		def fill(self, *args, closed=True, **kwargs):
			"""Override fill so that line is closed by default"""
			return super().fill(closed=closed, *args, **kwargs)

		def plot(self, *args, **kwargs):
			"""Override plot so that line is closed by default"""
			lines = super().plot(*args, **kwargs)
			for line in lines:
				self._close_line(line)

		def _close_line(self, line):
			x, y = line.get_data()
			# FIXME: markers at x[0], y[0] get doubled-up
			if x[0] != x[-1]:
				x = np.append(x, x[0])
				y = np.append(y, y[0])
				line.set_data(x, y)

		def set_varlabels(self, labels):
			self.set_thetagrids(np.degrees(theta), labels)

		def _gen_axes_patch(self):
			# The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
			# in axes coordinates.
			if frame == 'circle':
				return Circle((0.5, 0.5), 0.5)
			elif frame == 'polygon':
				return RegularPolygon((0.5, 0.5), num_vars,
									  radius=.5, edgecolor="k")
			else:
				raise ValueError("Unknown value for 'frame': %s" % frame)

		def _gen_axes_spines(self):
			if frame == 'circle':
				return super()._gen_axes_spines()
			elif frame == 'polygon':
				# spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
				spine = Spine(axes=self,
							  spine_type='circle',
							  path=Path.unit_regular_polygon(num_vars))
				# unit_regular_polygon gives a polygon of radius 1 centered at
				# (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
				# 0.5) in axes coordinates.
				spine.set_transform(Affine2D().scale(0.5).translate(.5, .5)
									+ self.transAxes)
				return {'polar': spine}
			else:
				raise ValueError("Unknown value for 'frame': %s" % frame)

	register_projection(RadarAxes)
	return theta

def create_radar_plot(data, filename, agent_names, colors, upper_bound_performance, lower_bound_performance=None, base_threshold=0.0):

	if lower_bound_performance is None:
		if len(data["performances"]) <= 1:
			raise ValueError("Performance lower bound must be provided for only a single agent.")
		lower_bound_performance = []
		for id in range(len(data["performances"][0])):
			lower_bound_performance.append(min([perf[id] for perf in data["performances"]]))

	print(lower_bound_performance)
	lower_bound_performance = [a-base_threshold for a in lower_bound_performance]

	N = len(data["type_names"])
	theta = radar_factory(N, frame='polygon')
	spoke_labels = data["type_names"]

	fig, axs = plt.subplots(figsize=(10, 10), nrows=1, ncols=1,
							subplot_kw=dict(projection='radar'))
	fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.3, bottom=0.0)

	case_data = data["performances"]
	ax = axs
	# Plot the four cases from the example data on separate Axes
	# ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
	# plt.title("My Title", pad=20)
	fig.suptitle('Agent Performance for Different Teammate Types',
			horizontalalignment='center', color='black', weight='bold',
			size='large')
	 
	# ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
	# 			 	horizontalalignment='center', verticalalignment='center')
	for d, color in zip(case_data, colors):
		rescaled_d = [max(0, min(1, (val - lb + 0.0) / (ub - lb + 0.0))) for val, ub, lb in zip(d, upper_bound_performance, lower_bound_performance)]
		ax.plot(theta, rescaled_d, color=color)
		ax.fill(theta, rescaled_d, facecolor=color, alpha=0.25, label='_nolegend_')
	ax.set_varlabels(spoke_labels)

	# add legend relative to top-left plot
	labels = agent_names
	legend = ax.legend(labels, loc=(0.9, .95),
						  	labelspacing=0.1, fontsize='small')
	
	plt.tight_layout()
	plt.savefig(filename)