import random

from simulator_geof2.MapGenerators.Map import Map
from simulator_geof2.MapGenerators.MapGenerator import MapGenerator
# Simple map, agents on one side, rewards on the other
from simulator_geof2.MapGenerators.Obstacle import Obstacle


class Corners(MapGenerator):
    def generate(self):
        if random.random() > 0.5:
            map = Map([Obstacle([136, 552], 192, 448), Obstacle([1072, 672], 336, 176)],
                      [[random.randint(80, 180), 264]],
                      [[random.randint(300, 550), random.randint(500, 700)],
                       [random.randint(650, 850), random.randint(500, 700)],
                       [random.randint(930, 1230), random.randint(300, 500)]])
        else:
            map = Map([Obstacle([1144, 552], 192, 448), Obstacle([208, 672], 336, 176)],
                               [[random.randint(1000, 1200), 264]],
                               [[random.randint(750, 950), random.randint(500, 700)],
                                [random.randint(400, 650), random.randint(500, 700)],
                                [random.randint(50, 350), random.randint(300, 500)]])

        return map
