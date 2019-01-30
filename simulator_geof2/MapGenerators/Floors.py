import random

from simulator_geof2.MapGenerators import Map
from simulator_geof2.MapGenerators import MapGenerator
from simulator_geof2.MapGenerators import Obstacle


# High platform on one side of the map
class Floors(MapGenerator):
    def generate(self):
        rndm = random.random()
        if rndm > 0.67:
            map = Map([Obstacle([320, 500], 560, 40), Obstacle([960, 500], 560, 40)],
                                  [[200, 450] if random.random() > 0.5 else [1080, 450]],
                                  [[random.randint(100, 600) if random.random() > 0.5 else random.randint(700, 1100),
                                    random.randint(300, 450)],
                                   [random.randint(700, 1100) if random.random() > 0.5 else random.randint(100, 600),
                                    random.randint(600, 720)]])
            map.is_terminal = lambda positions: (any([pos[1] > 660 for pos in positions]) and len(map.rewards) > 0 and
                                                 map.rewards[-1][1] < 500) or len(map.rewards) == 0
        elif rndm > 0.33:
            map = Map([Obstacle([320, 500], 560, 40), Obstacle([960, 500], 560, 40)],
                              [[200, 450] if random.random() > 0.5 else [1080, 450]],
                              [[random.randint(100, 600), random.randint(300, 450)],
                               [random.randint(700, 1100), random.randint(300, 450)]])
            map.is_terminal = lambda positions: any([pos[1] > 660 for pos in positions]) or len(map.rewards) == 0
        else:
            map = Map([Obstacle([320, 500], 560, 40), Obstacle([960, 500], 560, 40)],
                [[200, 450] if random.random() > 0.5 else [1080, 450]],
                [[random.randint(700, 1100), random.randint(600, 750)]])

        return map
