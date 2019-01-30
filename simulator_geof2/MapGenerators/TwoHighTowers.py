import random

from simulator_geof2.MapGenerators import Map
from simulator_geof2.MapGenerators import MapGenerator
# Simple map, agents on one side, rewards on the other
from simulator_geof2.MapGenerators import Obstacle


class TwoHighTowers(MapGenerator):
    def generate(self):
        rndm = random.random()
        if rndm > 0.5:
            map = Map([Obstacle([400, 570], 80, 380), Obstacle([880, 650], 80, 300)],
                      [[random.randint(100, 200), 400], [random.randint(200, 300), 700]],
                      [[random.randint(500, 700), random.randint(350, 700)],
                       [random.randint(1000, 1100), random.randint(350, 700)],
                       [880, random.randint(150, 300)]])
        else:
            map = Map([Obstacle([880, 570], 80, 380), Obstacle([400, 650], 80, 300)],
                      [[random.randint(1100, 1200), 400], [random.randint(980, 1080), 700]],
                      [[random.randint(500, 700), random.randint(350, 700)],
                       [random.randint(100, 200), random.randint(350, 700)],
                       [400, random.randint(150, 300)]])

        return map
