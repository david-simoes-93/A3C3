import abc

import numpy as np
from gym import spaces
from scipy import stats

from simulator_kilobots.lib import SimpleVelocityControlKilobot, CornerQuad, CircularGradientLight
from simulator_kilobots.envs.kilobots_env import KilobotsEnv


class IndependentKilobotsEnv(KilobotsEnv):
    def __init__(self, **kwargs):
        super(IndependentKilobotsEnv, self).__init__(**kwargs)
        self.actions = [[1, 0], [1, 1], [1, -1], [0, 0]]

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

        return super(IndependentKilobotsEnv, self).step(None)

    def get_reward(self, state, action, new_state):
        target = state["light"]
        reward = 0
        for i in range(len(state["objects"])):
            pos1 = state["objects"][i]
            pos2 = new_state["objects"][i]
            prev_dist = np.sqrt((pos1[0] - target[0]) ** 2 + (pos1[1] - target[1]) ** 2)
            next_dist = np.sqrt((pos2[0] - target[0]) ** 2 + (pos2[1] - target[1]) ** 2)
            reward += prev_dist - next_dist
        # compute reward based on task and swarm state
        return reward * 100

    def has_finished(self, state, action):
        done = True

        target = state["light"]
        for i in range(len(state["objects"])):
            pos1 = state["objects"][i]
            dist = np.sqrt((pos1[0] - target[0]) ** 2 + (pos1[1] - target[1]) ** 2)
            done = done and dist < 0.2

        return done

    def get_observation(self):
        obses = []
        for i in range(len(self._kilobots)):
            my_state = self._kilobots[i].get_state()
            # print("i", i, my_state)
            my_obs = [0] * (3 * len(self._kilobots) + 3 * len(self._objects))
            my_obs[0] = my_state[0]
            my_obs[1] = my_state[1]
            my_obs[2] = my_state[2]
            index = 3
            for j in range(len(self._kilobots)):
                if j == i:
                    continue
                # print("j", j, self._kilobots[j].get_state())
                other_state = self._kilobots[j].get_state()
                my_obs[index] = other_state[0] - my_obs[0]
                my_obs[index + 1] = other_state[1] - my_obs[1]
                my_obs[index + 2] = normalize_radian(other_state[2] - my_obs[2])
                index += 3
            for j in range(len(self._objects)):
                # print("j", j, self._objects[j].get_state())
                other_state = self._objects[j].get_state()
                my_obs[index] = other_state[0] - my_obs[0]
                my_obs[index + 1] = other_state[1] - my_obs[1]
                my_obs[index + 2] = normalize_radian(other_state[2] - my_obs[2])
                index += 3
                # my_obs.extend(self._objects[j].get_state() - my_state)
            obses.append(my_obs)

        return obses  # [self.get_state() for _ in range(self._kilobots)]

    def _configure_environment(self):
        # sample swarm spawn location
        self._swarm_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(.9, 1.4))
        swarm_spawn_location = self._swarm_spawn_distribution.rvs()

        # create objects
        self._objects = [
            CornerQuad(world=self.world, width=.15, height=.15, position=(-.75, 0), orientation=-np.pi / 2),
            # CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .605), orientation=-np.pi / 2),
            # CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .45), orientation=-np.pi)
        ]

        # create light
        self._light = CircularGradientLight(position=(.75, 0))  # swarm_spawn_location
        # self._lights = [GradientLight(np.array([0, .75]), np.array([0, -.75]))]

        # create kilobots
        self._kilobots = [SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.0, .0)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.03, .0)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.0, .03)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (-.03, .0))]


def normalize_radian(angle):
    if angle < -np.pi:
        angle += 2 * np.pi
    elif angle > np.pi:
        angle -= 2 * np.pi
    return angle
