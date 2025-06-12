from world.entities.station import Station
from pandas import DataFrame

class Replenishment(Station):
    def __init__(self, id: int, x: int, y: int, data: DataFrame, max_robots: int):
        super().__init__(id, "replenishment", x, y, data)
        self.max_robots = max_robots