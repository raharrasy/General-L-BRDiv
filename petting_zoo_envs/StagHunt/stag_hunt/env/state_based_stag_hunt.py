import functools
import logging
from collections import namedtuple
from enum import IntEnum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
from stag_hunt.agents.heuristic_agent import H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15
import numpy as np
from gymnasium.spaces import Box, Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

class Action(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.reward = 0
        self.history = None

        self.active = False

    def setup(self, position):
        self.history = []
        self.position = position
        self.reward = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
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
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"], "name": "staghunt_v1"}

    action_set = [Action.NONE, Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
    Observation = namedtuple(
        "Observation",
        ["actions", "players", "game_over", "episode_over", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        world_length=5,
        world_height=5,
        max_episode_steps=40,
        players=3,
        total_episodes = 10,
        seed = 1235,
        teammate_id = -1,
        blinded = True,
        render_mode=None,
    ):
        self.possible_agents = ["player_" + str(r) for r in range(players)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.logger = logging.getLogger(__name__)
        self.world_length = world_length
        self.world_height = world_height
        self.seed_val = seed
        self.viewer = None
        self.seed(seed)
        self.players = [Player() for _ in range(players)]
        self._max_episode_steps = max_episode_steps
        self._game_over = None
        self._episode_over = None
        self._rendering_initialized = False
        self.teammate_id = teammate_id
        self.blinded = blinded
        self.agents = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._get_observation_space()
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
    
    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y)*player_count
        """

        min_obs = [0, 0] * len(self.players)
        max_obs = [self.world_length-1, self.world_height-1] * len(self.players)

        return Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)

    @property
    def game_over(self):
        return self._game_over
    
    @property
    def episode_over(self):
        return self._episode_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def spawn_players(self):

        for player in self.players:
            if not player.active:
                continue

            player.reward = 0
            player.position = (
                self.np_random.choice(list(range(1,self.world_length-1)), 1),
                self.np_random.choice(list(range(1, self.world_height - 1)), 1)
            )

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.LEFT:
            return (
                player.position[0] > 0
            )
        elif action == Action.RIGHT:
            return (
                player.position[0] < self.world_length - 1
            )
        elif action == Action.UP:
            return (
                    player.position[1] > 0
            )
        elif action == Action.DOWN:
            return (
                    player.position[1] < self.world_height - 1
            )

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        if not player.active:
            return None

        return self.Observation(
                actions=self._valid_actions[player],
                players=[
                    self.PlayerObservation(
                        position=a.position,
                        is_self=a == player,
                        history=a.history,
                        reward=a.reward if a == player else None,
                    ) for a in self.players
                ],
                # todo also check max?
                game_over=self.game_over,
                episode_over=self.episode_over,
                current_step=self.current_step,
        )


    def _make_gym_obs(self, observations):
        def make_obs_array(observation):
            obs = -np.ones(self._get_observation_space().shape)
            if observation is None:
                return obs
            
            if not self.blinded:
                seen_players = [p for p in observation.players if p and p.is_self] + [
                     p for p in observation.players if (p and not p.is_self) or (not p)
                ]
            else:
                seen_players = [p for p in observation.players if p and p.is_self]

            for i in range(len(self.players)):
                obs[2*i] = -1
                obs[2*i+1] = -1

            for i, p in enumerate(seen_players):
                if p:
                    obs[2*i] = p.position[0]
                    obs[2*i+1]= p.position[1]
            return obs

        def get_player_reward(observation):
            if observation is None:
                return 0.0
            for p in observation.players:
                if p and p.is_self:
                    return p.reward


        nobs = {agent: make_obs_array(ob) if self.players[self.agent_name_mapping[agent]].active else make_obs_array(None)
                for agent, ob in zip(self.possible_agents, observations)}
        nreward = {agent: get_player_reward(obs) for agent, obs in zip(self.possible_agents, observations)}
        ndone = {agent: obs.game_over if obs else True for agent, obs in zip(self.possible_agents, observations)}
        ntruncated = {agent: obs.episode_over if obs else True for agent, obs in zip(self.possible_agents, observations)}
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {agent: {} for agent in self.possible_agents}

        # todo this?:
        # return nobs, nreward, ndone, ninfo
        # use this line to enable heuristic agents:
        return nobs, nreward, ndone, ntruncated, ninfo

    def _make_gym_obs_returns(self, observations):
        def get_player_reward(observation):
            if observation is None:
                return 0.0
            for p in observation.players:
                if p.is_self:
                    return p.reward

        nreward = [get_player_reward(obs) for obs in observations]
        return nreward

    def reset(self, seed=None):

        if seed != None:
            self.seed_val = seed
        elif self.seed_val != None:
            self.seed_val = self.seed_val + 123
        else:
            self.seed_val = 0
        self.seed(self.seed_val)
        self.agents = self.possible_agents[:]

        for idx in range(len(self.players)):
            self.players[idx].active = True

        self.spawn_players()

        self.current_step = 0
        self._game_over = False
        self._episode_over = False
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, n_truncated, ninfo = self._make_gym_obs(observations)

        return nobs, ninfo

    def step(self, action):
        self.current_step += 1

        for p in self.players:
            p.reward = 0
        action_list = [a for a in action.values()]
        actions = action_list
        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions) if p.active
        ]

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                continue
            elif action == Action.LEFT:
                player.position = (player.position[0]-1, player.position[1])
            elif action == Action.RIGHT:
                player.position = (player.position[0]+1, player.position[1])
            elif action == Action.UP:
                player.position = (player.position[0], player.position[1]-1)
            elif action == Action.DOWN:
                player.position = (player.position[0], player.position[1]+1)

        all_agents_finished = all([player.position in [(0,0), (self.world_length-1, self.world_height-1), (0, self.world_height-1), (self.world_length-1, 0)] for player in self.players])
        self._game_over = all_agents_finished
        self._episode_over = self._max_episode_steps <= self.current_step
        
        self._gen_valid_moves()

        selected_solutions = []
        for player in self.players:
            if player.position == (0,0):
                selected_solutions.append(0)
            elif player.position == (self.world_length-1, 0):
                selected_solutions.append(1)
            elif player.position == (0, self.world_height-1):
                selected_solutions.append(2)
            else:
                selected_solutions.append(3)
        payoff_matrix = [
            [[2.0, -0.5, -0.5, -0.5], [1.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.5, 1.0], [-0.5, -0.5, -0.5, 2.0]],
            [[2.0, 1.0, 1.0, -0.5], [-0.5, 0.5, 1.0, -0.5], [-0.5, 1.0, 0.5, -0.5], [-0.5, 1.0, 1.0, 2.0]]
        ]

        collected_stag = None
        if all_agents_finished:
            collected_stag = all([a == 0 for a in selected_solutions]) or all([a == 3 for a in selected_solutions])
            per_item_collected = [0] * 4
            for i in range(len(selected_solutions)):
                per_item_collected[selected_solutions[i]] += 1

        for i,p in enumerate(self.players):
            if all_agents_finished:
                if collected_stag:
                    p.reward = 3
                else:
                    if selected_solutions[i] == 0 or selected_solutions[i] == 3:
                        # penalty for capturing stag alone
                        p.reward = -0.5
                    else:
                        # Divide hare capture reward by number of players that captured hare
                        p.reward = 1.0/per_item_collected[selected_solutions[i]]
            else:
                p.reward = 0

        observations_post_remove = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, n_truncated, ninfo = self._make_gym_obs(observations_post_remove)
        if self._game_over or self._episode_over:
            self.agents = []

        return nobs, nreward, ndone, n_truncated, ninfo

    def _init_render(self):
        # TO DO: Fix rendering for staghunt'
        from .rendering import Viewer

        self.viewer = Viewer((self.world_length, self.world_height))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()