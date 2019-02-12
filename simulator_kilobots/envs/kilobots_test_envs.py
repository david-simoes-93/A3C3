from .kilobots_env import KilobotsEnv

import numpy as np
from scipy import stats

from ..lib.body import CornerQuad, Triangle, LForm, CForm, TForm
from ..lib.kilobot import PhototaxisKilobot, SimplePhototaxisKilobot
from ..lib.light import CircularGradientLight


class QuadPushingEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., .5

    def __init__(self):
        # distribution for sampling swarm position
        self._swarm_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(.9, 1.4))
        # distribution for sampling the pushing object
        self._obj_spawn_distribution = stats.uniform(loc=(.05, -.7), scale=(.9, .65))

        super().__init__()


class QuadAssemblyKilobotsEnv(KilobotsEnv):
    def __init__(self):
        # distribution for sampling swarm position
        self._swarm_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(.9, 1.4))
        # distribution for sampling the pushing object
        self._obj_spawn_distribution = stats.uniform(loc=(.05, -.7), scale=(.9, .65))

        super().__init__()

    def _configure_environment(self):
        # sample swarm spawn location
        swarm_spawn_location = self._swarm_spawn_distribution.rvs()

        # sample object location
        obj_spawn_location = self._obj_spawn_distribution.rvs()

        # create objects
        self._objects = [
            CornerQuad(world=self.world, width=.15, height=.15, position=(.45, .605)),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .605), orientation=-np.pi / 2),
            CornerQuad(world=self.world, width=.15, height=.15, position=(.605, .45), orientation=-np.pi),
            CornerQuad(world=self.world, width=.15, height=.15, position=obj_spawn_location, orientation=-np.pi / 2)
        ]

        # create light
        self._light = CircularGradientLight(position=swarm_spawn_location)  # swarm_spawn_location
        # self._lights = [GradientLight(np.array([0, .75]), np.array([0, -.75]))]

        # create kilobots
        self._kilobots = [PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .0),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, .0),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .03),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, .0),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, -.03),
                                                  light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, -.03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, .03),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (-.03, .0),
                                            light=self._light),
                          PhototaxisKilobot(self.world, position=swarm_spawn_location + (.0, -.03),
                                            light=self._light)
                          ]

    # TODO implement has_finished function
    def has_finished(self, state, action):
        return False

    # TODO implement reward function
    def get_reward(self, state, action, new_state):
        # compute reward based on task and swarm state
        return 1.

    # info function
    def get_info(self, state, action):
        return None


class TriangleTestEnv(KilobotsEnv):
    def _configure_environment(self):
        self._objects = [Triangle(world=self.world, width=.15, height=.15, position=(.0, .0)),
                         LForm(world=self.world, width=.15, height=.15, position=(.0, .3)),
                         TForm(world=self.world, width=.15, height=.15, position=(.0, -.3)),
                         CForm(world=self.world, width=.15, height=.15, position=(.3, .0))]

    # TODO implement has_finished function
    def has_finished(self, state, action):
        return False

    # TODO implement reward function
    def get_reward(self, state, action, new_state):
        # compute reward based on task and swarm state
        return 1.

    # info function
    def get_info(self, state, action):
        return None