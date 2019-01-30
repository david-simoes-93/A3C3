import random

from simulator_geof2.MapGenerators import Map
from simulator_geof2.MapGenerators import MapGenerator
from simulator_geof2.MapGenerators import Obstacle


# The Dome, circle can only get his reward with rectangle's help, 1 reward each
class Dome(MapGenerator):
    def generate(self):
        map = Map([Obstacle([400, 520], 80, 360), Obstacle([880, 520], 80, 360),
                   Obstacle([640, 360], 400, 40)],
                  [[random.randint(100, 300), 700], [random.randint(900, 1100), 700]],
                  [[random.randint(480, 800), random.randint(600, 700)],
                   [random.randint(500, 780), random.randint(200, 300)]])

        return map
