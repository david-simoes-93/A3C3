import abc
import time

import gym
import numpy as np
from Box2D import b2World, b2ChainShape

from ..lib.body import Body, _world_scale
from ..lib.kilobot import Kilobot
from ..lib.light import Light


class KilobotsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    world_size = world_width, world_height = 2., 1.5
    screen_size = screen_width, screen_height = 1200, 900

    _observe_objects = False
    _observe_light = True

    __sim_steps_per_second = 10
    __sim_velocity_iterations = 10
    __sim_position_iterations = 10
    __steps_per_action = 10

    def __new__(cls, **kwargs):
        cls.sim_steps_per_second = cls.__sim_steps_per_second
        cls.sim_step = 1. / cls.__sim_steps_per_second
        cls.world_x_range = -cls.world_width / 2, cls.world_width / 2
        cls.world_y_range = -cls.world_height / 2, cls.world_height / 2
        cls.world_bounds = (np.array([-cls.world_width / 2, -cls.world_height / 2]),
                            np.array([cls.world_width / 2, cls.world_height / 2]))

        return super(KilobotsEnv, cls).__new__(cls)

    def __init__(self, **kwargs):
        self.__sim_steps = 0
        self.__reset_counter = 0

        # create the Kilobots world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.table = self.world.CreateStaticBody(position=(.0, .0))
        self.table.CreateFixture(
            shape=b2ChainShape(vertices=[(_world_scale * self.world_x_range[0], _world_scale * self.world_y_range[1]),
                                         (_world_scale * self.world_x_range[0], _world_scale * self.world_y_range[0]),
                                         (_world_scale * self.world_x_range[1], _world_scale * self.world_y_range[0]),
                                         (_world_scale * self.world_x_range[1], _world_scale * self.world_y_range[1])]))
        self._real_time = False

        # add kilobots
        self._kilobots = []
        # add objects
        self._objects = []
        # add light
        self._ligh = None

        self.__seed = 0

        self._screen = None
        self.render_mode = 'human'
        self.video_path = None

        self._configure_environment()

        self._step_world()

    @property
    def _sim_steps(self):
        return self.__sim_steps

    @property
    def kilobots(self):
        return tuple(self._kilobots)

    @property
    def num_kilobots(self):
        return len(self._kilobots)

    @property
    def objects(self):
        return tuple(self._objects)

    @property
    def action_space(self):
        if self._light:
            return self._light.action_space

    @property
    def observation_space(self):
        return NotImplemented

    @property
    def state_space(self):
        return NotImplemented

    @property
    def _steps_per_action(self):
        return self.__steps_per_action

    def _add_kilobot(self, kilobot: Kilobot):
        self._kilobots.append(kilobot)

    def _add_object(self, body: Body):
        self._objects.append(body)

    @abc.abstractmethod
    def _configure_environment(self):
        raise NotImplementedError

    def get_state(self):
        return {'kilobots': np.array([k.get_state() for k in self._kilobots]),
                'objects': np.array([o.get_state() for o in self._objects]),
                'light': self._light.get_state()}

    def get_observation(self):
        return self.get_state()

    @abc.abstractmethod
    def get_reward(self, state, action, new_state):
        raise NotImplementedError

    def has_finished(self, state, action):
        return False

    def get_info(self, state, action):
        return ""

    def destroy(self):
        del self._objects[:]
        del self._kilobots[:]
        del self._light
        self._light = None
        if self._screen is not None:
            del self._screen
            self._screen = None

    def close(self):
        self.destroy()

    def seed(self, seed=None):
        if seed is not None:
            self.__seed = seed
        return [self.__seed]

    def reset(self):
        self.__reset_counter += 1
        self.destroy()
        self._configure_environment()
        self.__sim_steps = 0

        # step to resolve
        self._step_world()

        return self.get_observation()

    def step(self, action: np.ndarray):
        # if self.action_space and action is not None:
        #     assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # state before action is applied
        state = self.get_state()

        for i in range(self.__steps_per_action):
            _t_step_start = time.time()
            # step light
            if action is not None and self._light:
                self._light.step(action, self.sim_step)

            if self._light:
                # compute light values and gradients
                sensor_positions = np.array([kb.light_sensor_pos() for kb in self._kilobots])
                values, gradients = self._light.value_and_gradients(sensor_positions)

                for kb, v, g in zip(self._kilobots, values, gradients):
                    kb.set_light_value_and_gradient(v, g)

            # step kilobots
            for k in self._kilobots:
                k.step(self.sim_step)

            # step world
            self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
            self.world.ClearForces()

            self.__sim_steps += 1

            if self._screen is not None:
                self.render(self.render_mode)

            _t_step_end = time.time()

            if self._real_time:
                time.sleep(max(self.sim_step - (_t_step_end - _t_step_start), .0))

        # state
        next_state = self.get_state()

        # observation
        observation = self.get_observation()

        # reward
        reward = self.get_reward(state, action, next_state)

        # done
        done = self.has_finished(next_state, action)

        # info
        info = self.get_info(next_state, action)

        return observation, reward, done, info

    def _step_world(self):
        self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
        self.world.ClearForces()

    def render(self, mode=None):
        # if close:
        #     if self._screen is not None:
        #         self._screen.close()
        #         self._screen = None
        #     return
        if mode is None:
            mode = self.render_mode

        from simulator_kilobots.lib import kb_rendering
        if self._screen is None:
            caption = self.spec.id if self.spec else ""
            if self.video_path:
                import os
                os.makedirs(self.video_path, exist_ok=True)
                _video_path = os.path.join(self.video_path, str(self.__reset_counter) + '.mp4')
            else:
                _video_path = None

            self._screen = kb_rendering.KilobotsViewer(self.screen_width, self.screen_height, caption=caption,
                                                       display=mode == 'human', record_to=_video_path)
            world_min, world_max = self.world_bounds
            self._screen.set_bounds(world_min[0], world_max[0], world_min[1], world_max[1])
        elif self._screen.close_requested():
            self._screen.close()
            self._screen = None
            # TODO how to handle this event?

        # render table
        x_min, x_max = self.world_x_range
        y_min, y_max = self.world_y_range
        self._screen.draw_polygon([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)],
                                  color=(255, 255, 255))
        self._screen.draw_polyline([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)],
                                   width=.003)

        # allow to draw on table
        self._draw_on_table(self._screen)

        # render objects
        for o in self._objects:
            o.draw(self._screen)

        # render kilobots
        for kb in self._kilobots:
            kb.draw(self._screen)

        # render light
        if self._light is not None:
            self._light.draw(self._screen)

        # allow to draw on top
        self._draw_on_top(self._screen)

        self._screen.render()

    def get_objects(self) -> [Body]:
        return self._objects

    def get_kilobots(self) -> [Kilobot]:
        return self._kilobots

    def get_light(self) -> Light:
        return self._light

    def _draw_on_table(self, screen):
        pass

    def _draw_on_top(self, screen):
        pass


class UnknownObjectException(Exception):
    pass


class UnknownLightTypeException(Exception):
    pass
