from abc import ABC, abstractmethod

from simulator_geof2.MapGenerators import Map


# Map gen class
class MapGenerator(ABC):

    # Returns a Map object
    @abstractmethod
    def generate(self):
        return Map()
