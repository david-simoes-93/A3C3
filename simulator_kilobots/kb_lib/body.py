import abc

import Box2D
import numpy as np

_world_scale = 25.


class Body:
    _density = 2
    _friction = 0.01
    _restitution = 0.0

    _linear_damping = .8  # * _world_scale
    _angular_damping = .8  # * _world_scale

    def __init__(self, world: Box2D.b2World, position=None, orientation=None):
        if self.__class__ == Body:
            raise NotImplementedError('Abstract class Body cannot be instantiated.')
        self._color = np.array((93, 133, 195))
        self._highlight_color = np.array((238, 80, 62))

        if position is None:
            position = [.0, .0]
        position = np.asarray(position)

        if orientation is None:
            orientation = .0

        self._world = world
        self._body = world.CreateDynamicBody(
            position=Box2D.b2Vec2(*(_world_scale * position)),
            angle=orientation,
            linearDamping=self._linear_damping,
            angularDamping=self._angular_damping)

        self._body.linearVelocity = Box2D.b2Vec2(*[.0, .0])

        self._body.angularVelocity = .0

    @property
    def width(self):
        raise NotImplementedError

    @property
    def height(self):
        raise NotImplementedError

    def __del__(self):
        self._world.DestroyBody(self._body)

    def get_position(self):
        return np.asarray(self._body.position) / _world_scale

    def set_position(self, position):
        self._body.position = position * _world_scale

    def get_orientation(self):
        return self._body.angle

    def set_orientation(self, orientation):
        self._body.angle = orientation

    def get_pose(self):
        position = np.asarray(self._body.position) / _world_scale
        return tuple((*position, self._body.angle))

    def set_pose(self, pose):
        self.set_position(pose[:2])
        self.set_orientation(pose[2])

    def get_state(self):
        return self.get_pose()
        # return tuple((*self._body.position, self._body.angle))

    def get_local_point(self, point):
        return np.asarray(self._body.GetLocalPoint(_world_scale * np.asarray(point))) / _world_scale

    def get_local_orientation(self, angle):
        return angle - self._body.angle

    def get_local_pose(self, pose):
        return tuple((*self.get_local_point(pose[:2]), self.get_local_orientation(pose[2])))

    def get_world_point(self, point):
        return np.asarray(self._body.GetWorldPoint(_world_scale * np.asarray(point))) / _world_scale

    def collides_with(self, other):
        for contact_edge in self._body.contacts_gen:
            if contact_edge.other == other and contact_edge.contact.touching:
                return True

    # def set_color(self, color):
    #     self._color = color
    #
    # def set_highlight_color(self, color):
    #     self._highlight_color = color

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        color = np.asarray(color, dtype=np.int32)
        color = np.maximum(color, np.zeros_like(color, dtype=np.int32))
        color = np.minimum(color, np.full_like(color, 255, dtype=np.int32))
        self._color = color

    @property
    def highlight_color(self):
        return self._highlight_color

    @highlight_color.setter
    def highlight_color(self, color):
        color = np.asarray(color, dtype=np.int32)
        color = np.maximum(color, np.zeros_like(color, dtype=np.int32))
        color = np.minimum(color, np.full_like(color, 255, dtype=np.int32))
        self._highlight_color = color

    @abc.abstractmethod
    def draw(self, viewer):
        raise NotImplementedError('The draw method needs to be implemented by the subclass of Body.')

    @abc.abstractmethod
    def plot(self, axes, **kwargs):
        raise NotImplementedError('The plot method needs to be implemented by the subclass of Body.')


class Quad(Body):
    def __init__(self, width, height, **kwargs):
        super().__init__(**kwargs)

        self._width = width
        self._height = height

        self._fixture = self._body.CreatePolygonFixture(
            box=Box2D.b2Vec2(self._width/2 * _world_scale, self._height/2 * _world_scale),
            density=self._density,
            friction=self._friction,
            restitution=self._restitution,
            # radius=.000001
        )

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def vertices(self):
        return np.asarray([[self._body.GetWorldPoint(v) for v in self._fixture.shape.vertices]]) / _world_scale

    def draw(self, viewer):
        viewer.draw_polygon(self.vertices[0], filled=True, color=self._color)

    def plot(self, axes, **kwargs):
        from simulator_kilobots.kb_lib.kb_plotting import plot_rect
        return plot_rect(axes, self, **kwargs)

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height


class CornerQuad(Quad):
    def draw(self, viewer):
        super(CornerQuad, self).draw(viewer)

        viewer.draw_polygon(self.vertices[0][0:3], filled=True, color=self._highlight_color)

    def plot(self, axes, **kwargs):
        from simulator_kilobots.kb_lib.kb_plotting import plot_rect
        return plot_rect(axes, self, highlight_corner=True, **kwargs)


class Circle(Body):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)

        self._radius = radius

        self._fixture = self._body.CreateCircleFixture(
            radius=self._radius * _world_scale,
            density=self._density,
            friction=self._friction,
            restitution=self._restitution
        )

    @property
    def width(self):
        return 2 * self._radius

    @property
    def height(self):
        return 2 * self._radius

    def draw(self, viewer):
        viewer.draw_aacircle(position=self.get_position(), radius=self._radius, color=self._color)

    @property
    def vertices(self):
        return np.array([[self.get_position()]])

    def get_radius(self):
        return self._radius

    def plot(self, axes, **kwargs):
        from simulator_kilobots.kb_lib.kb_plotting import plot_circle
        return plot_circle(axes, self, **kwargs)


class Polygon(Body):
    def __init__(self, width: float, height: float, **kwargs):
        super().__init__(**kwargs)

        self._width = width
        self._height = height

        # TODO: right now this assumes that all subpolygons have the same number of edges
        # TODO: rewrite such that arbitrary subpolygons can be used here
        vertices = self._shape_vertices()

        v_size = np.amax(vertices, (0, 1)) - np.amin(vertices, (0, 1))
        vertices /= v_size
        vertices *= np.array((width, height))

        centroid = np.zeros(2)
        area = .0
        for vs in vertices:
            # compute centroid of polygon
            a = 0.5 * np.abs(np.dot(vs[:, 0], np.roll(vs[:, 1], 1)) - np.dot(vs[:, 1], np.roll(vs[:, 0], 1)))
            area += a
            centroid += vs.mean(axis=0) * a
        centroid /= area

        self.__local_vertices = vertices - centroid
        self.__local_vertices.setflags(write=False)

        for v in self.__local_vertices:
            self._body.CreatePolygonFixture(
                shape=Box2D.b2PolygonShape(vertices=(v * _world_scale).tolist()),
                density=self._density,
                friction=self._friction,
                restitution=self._restitution,
                # radius=.00000001
            )

        self._fixture = self._body.fixtures

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def vertices(self):
        return np.array([[self.get_world_point(v) for v in vertices] for vertices in self.__local_vertices])

    @property
    def local_vertices(self):
        return self.__local_vertices

    @property
    def plot_vertices(self):
        raise NotImplementedError

    @staticmethod
    def _shape_vertices() -> np.ndarray:
        raise NotImplementedError

    def draw(self, viewer):
        for vertices in self.vertices:
            viewer.draw_polygon(vertices, filled=True, color=self._color)

    def plot(self, axes, **kwargs):
        from simulator_kilobots.kb_lib.kb_plotting import plot_polygon
        return plot_polygon(axes, self, **kwargs)


class Triangle(Polygon):
    @staticmethod
    def _shape_vertices():
        return np.array([[(-0.5, 0.0),
                          (0.0, 0.0),
                          (0.0, 1.0)]])

    @property
    def plot_vertices(self):
        return self.vertices.reshape((-1, 2))


class LForm(Polygon):
    @staticmethod
    def _shape_vertices():
        return np.array([[(-0.05, 0.0), (0.1, 0.0), (0.1, 0.3), (-0.05, 0.3)],
                         [(0.1, 0.0), (0.1, -0.15), (-0.2, -0.15), (-0.2, 0.0)]])

    @property
    def plot_vertices(self):
        vertices = self.vertices.reshape((-1, 2))
        return vertices[[0, 7, 6, 5, 2, 3], :]


class TForm(Polygon):
    @staticmethod
    def _shape_vertices():
        return np.array([[(0.0, 0.15), (0.2, 0.15), (0.2, -0.15), (0.0, -0.15)],
                         [(0.0, 0.05), (0.0, -0.05), (-0.2, -0.05), (-0.2, 0.05)]])

    @property
    def plot_vertices(self):
        vertices = self.vertices.reshape((-1, 2))
        return vertices[[0, 1, 2, 3, 5, 6, 7, 4], :]


class CForm(Polygon):
    @staticmethod
    def _shape_vertices():
        return np.array([[(0.09, 0.15), (0.09, -0.15), (-0.01, -0.15), (-0.01, 0.15,)],
                         [(-0.01, -0.15), (-0.11, -0.15), (-0.11, -0.08), (-0.01, -0.05)],
                         [(-0.01, 0.15), (-0.11, 0.15), (-0.11, 0.08), (-0.01, 0.05)]])

    @property
    def plot_vertices(self):
        vertices = self.vertices.reshape((-1, 2))
        return vertices[[0, 1, 5, 6, 7, 11, 10, 9], :]
