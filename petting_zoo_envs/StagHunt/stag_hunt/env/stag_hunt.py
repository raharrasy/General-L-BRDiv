import functools
import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, MultiBinary

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

STAG = 0
HARE = 1
MOVES = ["STAG", "HARE"]
NUM_ITERS = 1


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    # TODO Set metadata and name
    metadata = {"render_modes": ["human"], "name": "stag_hunt_v1"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        # TODO Define num of agents
        self.possible_agents = ["player_" + str(r) for r in range(3)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    # TODO 
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiBinary(9)

    # TODO
    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}, Agent3: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]], MOVES[self.state[self.agents[2]]]
            )
        else:
            string = "Game over"
        print(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        none_obs = np.zeros(9)
        none_obs[NONE] = 1
        none_obs[NONE+3] = 1
        none_obs[NONE+3] = 1
        observations = {agent: none_obs for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = {agent: np.argmax(observations[agent]) for agent in observations}

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        # TODO
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        int_actions = [int(act) for act in actions.values()]
        
        if int_actions.count(int_actions[0]) == len(int_actions) and int_actions[0] == 0:
            for agent in self.agents:
                rewards[agent] = 2
        else:
            for agent in self.agents:
                rewards[agent] = 1 if actions[agent] == 1 else 0

        terminations = {agent: True for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # current observation is just the other player's most recent action
        
        observations = {}
        for i in range(len(self.agents)):
            obs_vec = np.zeros(9)
            added = 0
            for j in range(len(self.agents)):
                obs_vec[int(actions[self.agents[(i+1)%len(self.agents)]])+3*added] = 1
                added += 1
            observations[self.agents[i]] = obs_vec
        self.state = {agent: np.argmax(observations[agent]) for agent in observations}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation or all(terminations.values()):
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos
