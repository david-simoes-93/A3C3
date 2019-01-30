import random

from simulator_geof2.MapGenerators import Map
from simulator_geof2.MapGenerators import MapGenerator
from simulator_geof2.MapGenerators import Obstacle


# High platform on one side of the map
class TwoFloors(MapGenerator):
    def generate(self):
        rndm = random.random()
        if rndm > 0.5:
            map = Map([Obstacle([200, 500], 320, 40), Obstacle([840, 500], 800, 40),
                       Obstacle([440, 260], 800, 40), Obstacle([1080, 260], 320, 40)],
                      [[200, 160] if random.random() > 0.5 else [1080, 160]],
                      [[1000, 100] if random.random() > 0.5 else [200, 100],
                       [1000, 380] if random.random() > 0.5 else [200, 380],
                       [1000, 600] if random.random() > 0.5 else [200, 600]])
        else:
            map = Map([Obstacle([200, 260], 320, 40), Obstacle([840, 260], 800, 40),
                       Obstacle([440, 500], 800, 40), Obstacle([1080, 500], 320, 40)],
                      [[200, 160] if random.random() > 0.5 else [1080, 160]],
                      [[1000, 100] if random.random() > 0.5 else [200, 100],
                       [1000, 380] if random.random() > 0.5 else [200, 380],
                       [1000, 600] if random.random() > 0.5 else [200, 600]])

        map.is_terminal = lambda positions: (any([pos[1] > 660 for pos in positions]) and len(map.rewards) > 0 and
                                             map.rewards[-1][1] < 500) or len(map.rewards) == 0

        return map
