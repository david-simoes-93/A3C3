import abc

import numpy as np
from gym import spaces
from scipy import stats

from simulator_kilobots.kb_lib import SimpleVelocityControlKilobot, CornerQuad, CircularGradientLight, Triangle, Circle
from simulator_kilobots.envs.kilobots_env import KilobotsEnv


class IndependentKilobotsJoinEnv(KilobotsEnv):
    def __init__(self, **kwargs):
        self.number_of_agents = kwargs["number_of_agents"]
        super(IndependentKilobotsJoinEnv, self).__init__(**kwargs)
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

        return super(IndependentKilobotsJoinEnv, self).step(None)

    def get_reward(self, state, action, new_state):

        dist, dist_new = 0, 0
        for i in range(len(state["objects"])):
            pos1 = state["objects"][i]
            pos1_new = new_state["objects"][i]
            for j in range(len(state["objects"])):
                pos2 = state["objects"][j]
                pos2_new = new_state["objects"][j]
                dist += np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                dist_new += np.sqrt((pos1_new[0] - pos2_new[0]) ** 2 + (pos1_new[1] - pos2_new[1]) ** 2)

        # compute reward based on task and swarm state
        return (dist - dist_new) * 10

    def has_finished(self, state, action):
        done = True

        for i in range(len(state["objects"])):
            pos1 = state["objects"][i]
            for j in range(len(state["objects"])):
                pos2 = state["objects"][j]
                dist = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                done = done and dist < 0.25

        return done

    def get_observation(self):
        obses = []
        for i in range(len(self._kilobots)):
            my_state = self._kilobots[i].get_state()

            my_obs = [my_state[0], my_state[1], np.math.sin(my_state[2]), np.math.cos(my_state[2])]

            for j in range(len(self._objects)):
                my_obs.extend(get_polar(my_state, self._objects[j].get_state()))

            obses.append(my_obs)

        return obses  # [self.get_state() for _ in range(self._kilobots)]

    def _configure_environment(self):
        # sample swarm spawn location
        self._swarm_spawn_distribution = stats.uniform(loc=(0, 0), scale=(0.1, 0.1))
        swarm_spawn_location = self._swarm_spawn_distribution.rvs()

        # create objects
        self._objects = [
            CornerQuad(world=self.world, width=.15, height=.15, position=(-.605, 0), orientation=-np.pi / 2),
            Triangle(world=self.world, width=.15, height=.15, position=(.605, .605), orientation=-np.pi / 2),
            Circle(world=self.world, radius=.10, position=(.605, -.605))
        ]

        # create light
        self._light = CircularGradientLight(position=(2.75, 0))  # swarm_spawn_location
        # self._lights = [GradientLight(np.array([0, .75]), np.array([0, -.75]))]

        # create kilobots
        counter = 0
        curr_dist_mod = 1
        pos_mod_list = [(.06, .0), (-.06, .0), (.0, .06), (.0, -.06),
                        (.06, .06), (-.06, .06), (-.06, .06), (-.06, -.06)]
        self._kilobots = [SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.0, .0))]
        while len(self._kilobots) < self.number_of_agents:
            self._kilobots.append(
                SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location +
                                                                  (pos_mod_list[counter][0] * curr_dist_mod,
                                                                   pos_mod_list[counter][1] * curr_dist_mod)))
            counter = counter + 1
            if counter >= len(pos_mod_list):
                counter = 0
                curr_dist_mod += 1

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

# print(get_polar([1, 0, np.pi / 2], [1, -1]))
