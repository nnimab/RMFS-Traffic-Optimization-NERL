from __future__ import annotations
from typing import List, TYPE_CHECKING
from world.entities.zone import Zone
if TYPE_CHECKING:
    from world.warehouse import Warehouse
class ZoneManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.zones: List[Zone] = []
        self.zone_counter = 0
    
    def createZone(self, robots_location, warehouse_size, methods):
        obj = Zone(robots_location, warehouse_size, methods)
        self.zones.append(obj)
        self.zone_counter += 1
        return obj
    

