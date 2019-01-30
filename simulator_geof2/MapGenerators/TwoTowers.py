import random

from simulator_geof2.MapGenerators.Map import Map
from simulator_geof2.MapGenerators.MapGenerator import MapGenerator
# Simple map, agents on one side, rewards on the other
from simulator_geof2.MapGenerators.Obstacle import Obstacle


class TwoTowers(MapGenerator):
    def generate(self):
        rndm = random.random()
        if rndm > 0.67:
            map = Map([Obstacle([400, 620], 80, 280), Obstacle([880, 620], 80, 280)],
                      [[200, 700]],
                      [[random.randint(500, 700), random.randint(350, 700)] if random.random() > 0.5 else
                       [random.randint(900, 1100), random.randint(350, 700)],
                       [400, random.randint(150, 400)], [800, random.randint(150, 400)]])
        elif rndm > 0.33:
            map = Map([Obstacle([400, 620], 80, 280), Obstacle([880, 620], 80, 280)],
                      [[600, 700]],
                      [[random.randint(100, 300), random.randint(350, 700)] if random.random() > 0.5 else
                       [random.randint(900, 1100), random.randint(350, 700)],
                       [400, random.randint(150, 400)], [800, random.randint(150, 400)]])
        else:
            map = Map([Obstacle([400, 620], 80, 280), Obstacle([880, 620], 80, 280)],
                      [[1080, 700]],
                      [[random.randint(500, 700), random.randint(350, 700)] if random.random() > 0.5 else
                       [random.randint(100, 300), random.randint(350, 700)],
                       [400, random.randint(150, 400)], [800, random.randint(150, 400)]])

        return map
