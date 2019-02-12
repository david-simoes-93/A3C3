import abc

import numpy as np
from gym import spaces
from scipy import stats

from simulator_kilobots.lib import SimpleVelocityControlKilobot, CornerQuad, CircularGradientLight
from .kilobots_env import KilobotsEnv


class DirectControlKilobotsEnv(KilobotsEnv):
    def __init__(self, **kwargs):
        super(DirectControlKilobotsEnv, self).__init__(**kwargs)

    @property
    def action_space(self):
        as_low = np.array([kb.action_space.low for kb in self._kilobots])
        as_high = np.array([kb.action_space.high for kb in self._kilobots])
        return spaces.Box(as_low, as_high, dtype=np.float64)

    def step(self, actions: np.ndarray):
        if actions is not None:
            # assert self.action_space.contains(actions), 'actions not in action_space'

            for kb, a in zip(self.kilobots, actions):
                kb.set_action(a)

        else:
            for kb in self.kilobots:
                kb.set_action(None)

        return super(DirectControlKilobotsEnv, self).step(None)

    def get_reward(self, state, action, new_state):
        # compute reward based on task and swarm state
        return 1.

    def _configure_environment(self):
        # sample swarm spawn location
        self._swarm_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(.9, 1.4))
        swarm_spawn_location = self._swarm_spawn_distribution.rvs()

        # create objects
        self._objects = [
            CornerQuad(world=self.world, width=.15, height=.15, position=(.45, .605)),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .605), orientation=-np.pi / 2),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .45), orientation=-np.pi)
        ]

        # create light
        self._light = CircularGradientLight(position=swarm_spawn_location)  # swarm_spawn_location
        # self._lights = [GradientLight(np.array([0, .75]), np.array([0, -.75]))]

        # create kilobots
        self._kilobots = [SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.0, .0)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.03, .0)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (.0, .03)),
                          SimpleVelocityControlKilobot(self.world, position=swarm_spawn_location + (-.03, .0))]
