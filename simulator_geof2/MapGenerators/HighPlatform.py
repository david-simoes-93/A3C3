import random

from simulator_geof2.MapGenerators.Map import Map
from simulator_geof2.MapGenerators.MapGenerator import MapGenerator
from simulator_geof2.MapGenerators.Obstacle import Obstacle


# High platform on one side of the map
class HighPlatform(MapGenerator):
    def generate(self):
        if random.random() > 0.5:
            map = Map([Obstacle([480, 328], 960, 32)],
                      [[136, 264]],
                      [[136, 100], [1100, 272]])
        else:
            map = Map([Obstacle([800, 328], 960, 32)],
                      [[1150, 264]],
                      [[1150, 100], [180, 272]])

        map.is_terminal = lambda positions: any([pos[1] > 700 for pos in positions]) or len(map.rewards) == 0

        return map
