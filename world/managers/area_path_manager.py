from __future__ import annotations
from world.entities.area_path import AreaPath
from typing import List, TYPE_CHECKING
from lib.enum.area_path_type import AreaPathType
if TYPE_CHECKING:
    from world.warehouse import Warehouse

class AreaPathManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.area_paths: List[AreaPath] = []
        self.area_path_counter = 0

    def initAreaPathManager(self):
        for area_path in self.area_paths:
            area_path.setAreaPathManager(self)

    def getAllAreaPaths(self):
        return self.area_paths
    
    def createAreaPath(self, x: int, y: int, type_value: AreaPathType):
        obj = AreaPath(self.area_path_counter, x, y, type_value)
        self.area_paths.append(obj)
        self.area_path_counter += 1
        return obj