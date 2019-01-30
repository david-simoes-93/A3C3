import random

from simulator_geof2.MapGenerators import Map
from simulator_geof2.MapGenerators import MapGenerator
from simulator_geof2.MapGenerators import Obstacle


# split map, circle can only access top corners, rectangle only bottom corners, 1 reward each
class Split(MapGenerator):
    def generate(self):
        map = Map([Obstacle([400, 590], 80, 220), Obstacle([880, 590], 80, 220),
                   Obstacle([200, 500], 400, 40), Obstacle([1080, 500], 400, 40)],
                  [[random.randint(500, 700), 600], [random.randint(500, 700), 700]],
                  [[random.randint(100, 300), random.randint(600, 700)] if random.random() > 0.5 else
                   [random.randint(980, 1180), random.randint(600, 700)],
                   [random.randint(100, 300), random.randint(200, 400)] if random.random() > 0.5 else
                   [random.randint(980, 1180), random.randint(200, 400)]])

        return map
