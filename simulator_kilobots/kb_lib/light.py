import numpy as np

from typing import Iterable, Callable, Optional

from gym import spaces


class Light(object):
    relative_actions = True
    interpolate_actions = True

    def __init__(self, **kwargs):
        self.observation_space = None
        self.action_space = None
        
    def step(self, action, time_step: float):
        raise NotImplementedError

    def get_value(self, position: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_gradient(self, position: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def value_and_gradients(self, position: np.ndarray) -> (np.ndarray, np.ndarray):
        return self.get_value(position), self.get_gradient(position)

    def get_state(self):
        raise NotImplementedError

    def draw(self, viewer):
        raise NotImplementedError


class SinglePositionLight(Light):
    def __init__(self, *, position: np.ndarray = None, bounds: (np.ndarray, np.ndarray) = None,
                 action_bounds: (np.ndarray, np.ndarray) = None, relative_actions: bool = True, **kwargs):
        super().__init__(**kwargs)
        if position is None:
            self._position = np.array((.0, .0))
        else:
            self._position = position

        self._bounds = bounds
        if self._bounds is None:
            self._bounds = np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf])

        self._relative_actions = relative_actions
        self._action_bounds = action_bounds
        if self._action_bounds is None:
            if self._relative_actions:
                self._action_bounds = np.array([-0.01, -0.01]), np.array([.01, .01])
            else:
                self._action_bounds = self._bounds

        self.action_space = spaces.Box(*self._action_bounds, dtype=np.float64)
        self.observation_space = spaces.Box(*self._bounds, dtype=np.float64)

    def step(self, action: np.ndarray, time_step: float):
        if action is None:
            return

        action = action.squeeze()

        if self._action_bounds is not None:
            action = np.maximum(action, self._action_bounds[0])
            action = np.minimum(action, self._action_bounds[1])

        if self._relative_actions:
            self._position += action * time_step
        else:
            self._position = action

        self._position = np.maximum(self._position, self._bounds[0])
        self._position = np.minimum(self._position, self._bounds[1])

    def get_value(self, position: np.ndarray):
        return -1 * np.linalg.norm(position - self._position, axis=1)

    def get_gradient(self, position: np.ndarray):
        gradient = -1 * (position - self._position)
        return gradient / np.linalg.norm(gradient, axis=1)

    def value_and_gradients(self, position: np.ndarray):
        gradients = -1 * (position - self._position)
        gradient_norms = np.linalg.norm(gradients, axis=1)
        return -1 * gradient_norms, gradients / gradient_norms

    def get_position(self):
        return self._position

    def get_state(self):
        return self._position

    def draw(self, viewer):
        viewer.draw_aacircle(position=self._position, radius=.01, color=(255, 30, 30, 150))


class CompositeLight(Light):
    def __init__(self, lights: Iterable[Light] = None, reducer: Callable[[np.ndarray, Optional[int]], float] = np.sum):
        """

        :type lights: Iterable[Light] a list of Light objects
        :type reducer: Callable[[np.ndarray, Optional[int]], float]
        """
        super().__init__()
        self._lights = lights
        self._reducer = reducer

        self.observation_space = spaces.Box(np.concatenate(list(l.observation_space.low for l in self._lights)),
                                            np.concatenate(list(l.observation_space.high for l in self._lights)),
                                            dtype=np.float64)
        self.action_space = spaces.Box(np.concatenate(list(l.action_space.low for l in self._lights)),
                                       np.concatenate(list(l.action_space.high for l in self._lights)),
                                       dtype=np.float64)
        self._action_dims = list(l.action_space.shape[0] for l in self._lights)

    @property
    def lights(self):
        return tuple(self._lights)

    def step(self, action, time_step):
        if action is not None:
            action = action.squeeze()
            for l, ad in zip(self._lights, self._action_dims):
                l.step(action[:ad], time_step)
                action = action[ad:]

    def get_value(self, position: np.ndarray):
        return np.sum(np.array([l.get_value(position) for l in self._lights]), axis=0)

    def get_gradient(self, position: np.ndarray):
        max_l = np.argmax([l.get_value(position) for l in self._lights], axis=0)
        grads = np.array([l.get_gradient(position) for l in self._lights])
        return grads[max_l, range(grads.shape[1])].squeeze()

    def value_and_gradients(self, position: np.ndarray):
        values, grads = map(np.asarray, zip(*[l.value_and_gradients(position) for l in self._lights]))
        value = np.sum(values, axis=0)
        max_l = np.argmax(values, axis=0)
        return value, grads[max_l, range(position.shape[0])].squeeze()

    def get_state(self):
        return np.concatenate(list(l.get_state() for l in self._lights))

    def draw(self, viewer):
        for l in self._lights:
            l.draw(viewer)


class CircularGradientLight(SinglePositionLight):
    def __init__(self, radius=.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._radius = radius

    def get_value(self, position: np.ndarray):
        distance = np.linalg.norm(position - self._position)

        # compute value as linear interpolation between 255 and 0
        value = np.ones(position.shape[0])
        value -= distance / self._radius
        value = np.maximum(np.minimum(value, 1.), .0)
        value *= 255

        return value

    def get_gradient(self, position: np.ndarray):
        gradient = -1 * (position - self._position)
        norm_gradient = np.linalg.norm(gradient, axis=-1)

        gradient[norm_gradient <= self._radius] /= norm_gradient[norm_gradient < self._radius, None]
        gradient[norm_gradient > self._radius] *= .0

        return gradient

    def value_and_gradients(self, position: np.ndarray):
        gradient = -1 * (position - self._position)
        norm_gradient = np.linalg.norm(gradient, axis=1)

        # compute value as linear interpolation between 255 and 0
        value = np.ones(position.shape[0])
        value -= norm_gradient / self._radius
        value = np.maximum(np.minimum(value, 1.), .0)
        value *= 255
        # normalize gradients and set gradients to zero if norm larger than radius
        gradient /= norm_gradient[:, None]
        gradient[norm_gradient > self._radius] *= .0

        return value, gradient

    def get_state(self):
        return self._position

    def draw(self, viewer):
        viewer.draw_transparent_circle(position=self._position, radius=self._radius, color=(255, 255, 30, 150))


class SmoothGridLight(Light):
    def __init__(self):
        super(SmoothGridLight, self).__init__()

    def step(self, action, time_step):
        raise NotImplementedError

    def get_value(self, position: np.ndarray):
        raise NotImplementedError

    def get_gradient(self, position: np.ndarray):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def draw(self, viewer):
        raise NotImplementedError


class GradientLight(Light):
    relative_actions = False
    interpolate_actions = False

    def __init__(self, angle: float = .0):
        super().__init__()

        self._gradient_angle = np.array([angle])
        self._gradient_vec = np.r_[np.cos(angle), np.sin(angle)]

        self._bounds = np.array([-np.pi]), np.array([np.pi])
        if self.relative_actions:
            self._action_bounds = .5 * np.array([-np.pi]), .5 * np.array([np.pi])
        else:
            self._action_bounds = 2 * np.array([-np.pi]), 2 * np.array([np.pi])

        self.observation_space = spaces.Box(*self._bounds, dtype=np.float64)
        self.action_space = spaces.Box(*self._action_bounds, dtype=np.float64)

    def step(self, action, time_step):
        if action is None:
            return

        action = np.maximum(action, self._action_bounds[0])
        action = np.minimum(action, self._action_bounds[1])
        if self.relative_actions:
            self._gradient_angle += action * time_step
        else:
            self._gradient_angle = action

        if self._gradient_angle < self._bounds[0]:
            self._gradient_angle += 2 * np.pi
        if self._gradient_angle > self._bounds[1]:
            self._gradient_angle -= 2 * np.pi

        self._gradient_vec = np.r_[np.cos(self._gradient_angle), np.sin(self._gradient_angle)]

    def get_value(self, position: np.ndarray):
        projection = self._gradient_vec.dot(position)
        return projection

    def get_gradient(self, position: np.ndarray):
        return self._gradient_vec

    def get_state(self):
        return self._gradient_angle

    def set_angle(self, angle):
        self._gradient_angle = np.array([angle])
        self._gradient_vec = np.r_[np.cos(angle), np.sin(angle)]

    def draw(self, viewer):
        viewer.draw_polyline((np.array([0, 0]), self._gradient_vec), color=(1, 0, 0))
        pass


class MomentumLight(CircularGradientLight):
    interpolate_actions = False

    def __init__(self, velocity=None, max_velocity=None, action_bounds=None, **kwargs):
        super().__init__(**kwargs)

        self._action_bounds = action_bounds
        if self._action_bounds is None:
            self._action_bounds = np.array([-.01, -.01]), np.array([.01, .01])

        if velocity is None:
            self._velocity = np.array([.0, .0])
        else:
            self._velocity = velocity

        if max_velocity is None:
            self.max_velocity = np.inf
        else:
            self.max_velocity = max_velocity

        self._obs_bounds = np.r_[self._bounds[0], [-max_velocity, -max_velocity]], \
                           np.r_[self._bounds[1], [max_velocity, max_velocity]]

        self.action_space = spaces.Box(*self._action_bounds, dtype=np.float64)
        self.observation_space = spaces.Box(*self._obs_bounds, dtype=np.float64)

    def step(self, action: np.ndarray, time_step: float):
        if action is not None:
            action = action.squeeze()

            if self._action_bounds is not None:
                action = np.maximum(action, self._action_bounds[0])
                action = np.minimum(action, self._action_bounds[1])

            self._velocity += action * time_step

        if self.max_velocity is not None and np.linalg.norm(self._velocity) > self.max_velocity:
            self._velocity *= self.max_velocity / np.linalg.norm(self._velocity)

        self._position += self._velocity * time_step

        self._position = np.maximum(self._position, self._bounds[0])
        self._position = np.minimum(self._position, self._bounds[1])

    def get_state(self):
        return np.r_[self._position, self._velocity]
