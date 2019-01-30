import random

from simulator_geof2.MapGenerators.Map import Map
from simulator_geof2.MapGenerators.MapGenerator import MapGenerator


# Simple map, agents on one side, rewards on the other
class Basic(MapGenerator):

    def generate(self):
        if random.random() > 0.5:
            map = Map([],
                      [[random.randint(350, 600), 700], [random.randint(100, 300), 700]],
                      [[random.randint(700, 1150), random.randint(300, 700)]])
        else:
            map = Map([],
                      [[random.randint(950, 1150), 700], [random.randint(700, 900), 700]],
                      [[random.randint(100, 600), random.randint(300, 700)]])

        return map
