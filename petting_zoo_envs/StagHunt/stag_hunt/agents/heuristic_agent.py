import random
import numpy as np
from enum import IntEnum
import logging
import numpy as np

class Action(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

_MAX_INT = 999999

class H1(object):
    """
	H1 agent always goes to the closest rewarding goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        if self.target_goal == None:
            dist_to_goal = [abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1]) for goal in self.goals]
            min_dist_to_goal = min(dist_to_goal)
            closest_goals = [idx for idx, dist in enumerate(dist_to_goal) if dist==min_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H2(object):
    """
	H2 agent always goes to the furthest rewarding goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        if self.target_goal == None:
            dist_to_goal = [abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1]) for goal in self.goals]
            max_dist_to_goal = max(dist_to_goal)
            closest_goals = [idx for idx, dist in enumerate(dist_to_goal) if dist==max_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H3(object):
    """
	H3 agent always goes to the closest optimal goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        max_rewards = max(self.rewards)
        if self.target_goal == None:
            dist_to_goal = [(idx, abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1])) for idx, goal in enumerate(self.goals)]
            valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] == max_rewards]
            min_dist_to_goal = min([dist for _, dist in valid_dists])
            closest_goals = [idx for idx, dist in valid_dists if dist==min_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H4(object):
    """
	H4 agent always goes to the furthest optimal goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        max_rewards = max(self.rewards)
        if self.target_goal == None:
            dist_to_goal = [(idx, abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1])) for idx, goal in enumerate(self.goals)]
            valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] == max_rewards]
            max_dist_to_goal = max([dist for _, dist in valid_dists])
            closest_goals = [idx for idx, dist in valid_dists if dist==max_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H5(object):
    """
	H5 agent always goes to the closest suboptimal goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        max_rewards = max(self.rewards)
        if self.target_goal == None:
            dist_to_goal = [(idx, abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1])) for idx, goal in enumerate(self.goals)]
            valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] != max_rewards]
            min_dist_to_goal = min([dist for _, dist in valid_dists])
            closest_goals = [idx for idx, dist in valid_dists if dist==min_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H6(object):
    """
	H6 agent always goes to the furthest suboptimal goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        max_rewards = max(self.rewards)
        if self.target_goal == None:
            dist_to_goal = [(idx, abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1])) for idx, goal in enumerate(self.goals)]
            valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] != max_rewards]
            max_dist_to_goal = max([dist for _, dist in valid_dists])
            closest_goals = [idx for idx, dist in valid_dists if dist==max_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H7(object):
    """
	H7 goes to a randomly selected goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        if self.target_goal == None:
            self.target_goal = self.goals[np.random.choice(list(range(len(self.goals))), 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H8(object):
    """
	H8 goes to goal 1
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        self.target_goal = self.goals[0]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return


class H9(object):
    """
	H9 goes to goal 2
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        self.target_goal = self.goals[1]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H10(object):
    """
	H10 goes to goal 2
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        self.target_goal = self.goals[2]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H11(object):
    """
	H11 goes to goal 3
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        self.target_goal = self.goals[3]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        return

class H12(object):
    """
	H12 goes to a randomly selected goal, with goal 0 being preferred
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        if self.target_goal == None:
            self.target_goal = self.goals[np.random.choice(list(range(len(self.goals))), 1, p=[0.55, 0.15, 0.15, 0.15])[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        self.target_goal = None

class H13(object):
    """
	H13 goes to a randomly selected goal, with goal 1 being preferred
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        if self.target_goal == None:
            self.target_goal = self.goals[np.random.choice(list(range(len(self.goals))), 1, p=[0.15, 0.55, 0.15, 0.15])[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        self.target_goal = None

class H14(object):
    """
	H14 goes to a randomly selected goal, with goal 2 being preferred
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        if self.target_goal == None:
            self.target_goal = self.goals[np.random.choice(list(range(len(self.goals))), 1, p=[0.15, 0.15, 0.55, 0.15])[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        self.target_goal = None

class H15(object):
    """
	H15 goes to a randomly selected goal, with goal 3 being preferred
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.LEFT
        elif target_coord[1] < agent_pos[1]:
            return Action.UP
        elif target_coord[0] > agent_pos[0]:
            return Action.RIGHT
        elif target_coord[1] > agent_pos[1]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:2])
        if self.target_goal == None:
            self.target_goal = self.goals[np.random.choice(list(range(len(self.goals))), 1, p=[0.15, 0.15, 0.15, 0.55])[0]]

        return self.move_towards(self.target_goal, agent_pos)

    def reset(self):
        self.target_goal = None