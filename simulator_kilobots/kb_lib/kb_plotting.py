import math
import numpy as np
from simulator_kilobots.kb_lib import Body, Quad, CornerQuad, Circle
from simulator_kilobots.kb_lib.body import Polygon

from matplotlib.axes import Axes


def get_body_from_shape(object_shape, object_width, object_height, object_init):
    from simulator_kilobots.kb_lib import Quad, Triangle, Circle, LForm, TForm, CForm
    from Box2D import b2World

    fake_world = b2World()

    if object_shape.lower() in ['quad', 'rect', 'square']:
        return Quad(width=object_width, height=object_height,
                    position=object_init[:2], orientation=object_init[2],
                    world=fake_world)
    elif object_shape.lower() in ['corner_quad', 'corner-quad', 'corner_square', 'corner-square']:
        return CornerQuad(width=object_width, height=object_height,
                          position=object_init[:2], orientation=object_init[2],
                          world=fake_world)
    elif object_shape.lower() == 'triangle':
        return Triangle(width=object_width, height=object_height,
                        position=object_init[:2], orientation=object_init[2],
                        world=fake_world)
    elif object_shape.lower() == 'circle':
        return Circle(radius=object_width, position=object_init[:2],
                      orientation=object_init[2], world=fake_world)
    elif object_shape.lower() == 'l_shape':
        return LForm(width=object_width, height=object_height,
                     position=object_init[:2], orientation=object_init[2],
                     world=fake_world)
    elif object_shape.lower() == 't_shape':
        return TForm(width=object_width, height=object_height,
                     position=object_init[:2], orientation=object_init[2],
                     world=fake_world)
    elif object_shape.lower() == 'c_shape':
        return CForm(width=object_width, height=object_height,
                     position=object_init[:2], orientation=object_init[2],
                     world=fake_world)


def plot_body_from_shape(axes: Axes, object_shape: str, object_width: float, object_height: float, object_init,
                         **kwargs):
    body = get_body_from_shape(object_shape, object_width, object_height, object_init)
    return plot_body(axes, body, **kwargs), body


def plot_body(axes: Axes, body: Body, **kwargs):
    if isinstance(body, CornerQuad):
        return plot_rect(axes, body, highlight_corner=True, **kwargs)
    if isinstance(body, Quad):
        return plot_rect(axes, body, **kwargs)
    if isinstance(body, Circle):
        return plot_circle(axes, body, **kwargs)
    if isinstance(body, Polygon):
        return plot_polygon(axes, body, **kwargs)


def update_body(body, artist):
    if isinstance(body, Quad):
        update_rect(body, artist)
    if isinstance(body, Circle):
        update_circle(body, artist)
    if isinstance(body, Polygon):
        update_polygon(body, artist)


def plot_rect(axes: Axes, rect, highlight_corner=False, **kwargs):
    from matplotlib.patches import Rectangle, Polygon
    defaults = dict(fill=True, edgecolor='#929591', facecolor='#d8dcd6', highlight_fill=True,
                    highlight_facecolor='#d8dcd6')
    for k in defaults:
        if k not in kwargs:
            kwargs[k] = defaults[k]

    if 'alpha' in kwargs:
        from matplotlib.colors import to_rgba
        kwargs['edgecolor'] = to_rgba(kwargs['edgecolor'], kwargs['alpha'])
        kwargs['facecolor'] = to_rgba(kwargs['facecolor'], kwargs['alpha'])
        kwargs['highlight_facecolor'] = to_rgba(kwargs['highlight_facecolor'], kwargs['alpha'])

    highlight_facecolor = kwargs.pop('highlight_facecolor')
    highlight_fill = kwargs.pop('highlight_fill')

    x, y, theta = rect.get_pose()
    corner = rect.get_world_point((-rect.width / 2, -rect.height / 2))
    p1 = axes.add_patch(Rectangle(xy=corner, angle=math.degrees(theta),
                                  width=rect.width, height=rect.height, **kwargs))

    if highlight_corner:
        return p1, axes.add_patch(Polygon(xy=np.array(rect.vertices[0:3]), fill=highlight_fill,
                                          facecolor=highlight_facecolor))
    else:
        return p1


def update_rect(rect, artist):
    if isinstance(artist, tuple):
        update_rect(rect, artist[0])
        artist[1].set_xy(np.array(rect.vertices[0:3]))
    else:
        x, y, theta = rect.get_pose()
        corner = rect.get_world_point((-rect.width / 2, -rect.height / 2))
        artist.set_xy(corner)
        artist.angle = math.degrees(theta)


def plot_circle(axes: Axes, circle, **kwargs):
    from matplotlib.patches import Circle
    defaults = dict(fill=True, edgecolor='#929591', facecolor='#d8dcd6')
    for k in defaults:
        if k not in kwargs:
            kwargs[k] = defaults[k]

    return axes.add_patch(Circle(xy=circle.get_position(), radius=circle.get_radius(), **kwargs))


def update_circle(circle, artist):
    artist.set_center(circle.get_position())
    artist.set_radius(circle.get_radius())


def plot_polygon(axes: Axes, polygon, **kwargs):
    from matplotlib.patches import Polygon
    defaults = dict(fill=True,
                    # edgecolor='#929591',
                    facecolor='#d8dcd6')
    for k in defaults:
        if k not in kwargs:
            kwargs[k] = defaults[k]

    if 'alpha' in kwargs:
        from matplotlib.colors import to_rgba
        # kwargs['edgecolor'] = to_rgba(kwargs['edgecolor'], kwargs['alpha'])
        kwargs['facecolor'] = to_rgba(kwargs['facecolor'], kwargs['alpha'])

    artist = axes.add_patch(Polygon(xy=np.array(polygon.plot_vertices), **kwargs))

    return artist


def update_polygon(polygon, artist):
    artist.set_xy(np.array(polygon.plot_vertices))
