import abc

import numpy as np
from gym import spaces
from scipy import stats

from simulator_kilobots.kb_lib import SimpleVelocityControlKilobot, CornerQuad, CircularGradientLight, Triangle, Circle
from simulator_kilobots.envs.kilobots_env import KilobotsEnv


class IndependentKilobotsSplitEnv(KilobotsEnv):
    def __init__(self, **kwargs):
        super(IndependentKilobotsSplitEnv, self).__init__(**kwargs)
        self.actions = [[1, 0], [0, .1], [0, -.1], [0, 0]]

    @property
    def action_space(self):
        as_low = np.array([kb.action_space.low for kb in self._kilobots])
        as_high = np.array([kb.action_space.high for kb in self._kilobots])
        return spaces.Box(as_low, as_high, dtype=np.float64)

    def step(self, actions: np.ndarray):
        if actions is not None:
            # assert self.action_space.contains(actions), 'actions not in action_space'

            for kb, a in zip(self.kilobots, actions):
                kb.set_action(self.actions[a])

        else:
            for kb in self.kilobots:
                kb.set_action(None)

        return super(IndependentKilobotsSplitEnv, self).step(None)

    def get_reward(self, state, action, new_state):
        dist = get_dist(state["objects"][0], state["objects"][2]) + \
               get_dist(state["objects"][0], state["objects"][3]) + \
               get_dist(state["objects"][1], state["objects"][2]) + \
               get_dist(state["objects"][1], state["objects"][3])

        new_dist = get_dist(new_state["objects"][0], new_state["objects"][2]) + \
               get_dist(new_state["objects"][0], new_state["objects"][3]) + \
               get_dist(new_state["objects"][1], new_state["objects"][2]) + \
               get_dist(new_state["objects"][1], new_state["objects"][3])

        # compute reward based on task and swarm state
        return (new_dist - dist) * 10

    def has_finished(self, state, action):
        done = get_dist(state["objects"][0], state["objects"][2]) > 1 and \
               get_dist(state["objects"][0], state["objects"][3]) > 1 and \
               get_dist(state["objects"][1], state["objects"][2]) > 1 and \
               get_dist(state["objects"][1], state["objects"][3]) > 1

        return done

    def get_observation(self):
        obses = []
        for i in range(len(self._kilobots)):
            my_state = self._kilobots[i].get_state()

            my_obs = [my_state[0], my_state[1], np.math.sin(my_state[2]), np.math.cos(my_state[2])]
            for j in range(len(self._kilobots)):
                if j == i:
                    continue
                my_obs.extend(get_polar(my_state, self._kilobots[j].get_state()))
            for j in range(len(self._objects)):
                my_obs.extend(get_polar(my_state, self._objects[j].get_state()))

            obses.append(my_obs)

        return obses  # [self.get_state() for _ in range(self._kilobots)]

    def _configure_environment(self):
        # sample swarm spawn location
        self._swarm_spawn_distribution = stats.uniform(loc=(0, 0), scale=(0.5, 0.5))
        swarm_spawn_location = self._swarm_spawn_distribution.rvs()

        # create objects
        self._objects = [
            CornerQuad(world=self.world, width=.15, height=.15, position=(-.1, .1), orientation=-np.pi / 2),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.1, -.1), orientation=-np.pi / 2),
            Circle(world=self.world, radius=.08, position=(-.1, -.1)),
            Circle(world=self.world, radius=.08, position=(.1, .1))
        ]

        # create light
        self._light = CircularGradientLight(position=(2.75, 0))  # swarm_spawn_location
        # self._lights = [GradientLight(np.array([0, .75]), np.array([0, -.75]))]

        # create kilobots
        self._kilobots = [SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.03, -.03)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.03, .03)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (-.03, -.03)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (-.03, .03))]


def normalize_radian(angle):
    if angle < -np.pi:
        angle += 2 * np.pi
    elif angle > np.pi:
        angle -= 2 * np.pi
    return angle


# https://www.mathsisfun.com/polar-cartesian-coordinates.html
def get_polar(my_state, other_state):
    rel_pos = [other_state[0] - my_state[0], other_state[1] - my_state[1]]
    r = np.sqrt(rel_pos[0] * rel_pos[0] + rel_pos[1] * rel_pos[1])
    if rel_pos[0] != 0:
        angle = normalize_radian(np.arctan(rel_pos[1] / rel_pos[0]) - my_state[2])
    else:
        angle = normalize_radian((np.pi / 2 if rel_pos[1] > 0 else -np.pi / 2) - my_state[2])
    return [r, angle]


def get_dist(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

# print(get_polar([1, 0, np.pi / 2], [1, -1]))
