from typing import List, Optional, Dict, TYPE_CHECKING
from pandas import DataFrame
from world.entities.object import Object
from lib.types.netlogo_coordinate import NetLogoCoordinate
from .order import Order
if TYPE_CHECKING:
    from world.managers.station_manager import StationManager

class Station(Object):
    def __init__(self, id: int, station_type: str, x: int, y: int, data: DataFrame):
        super().__init__(id, station_type, x, y)
        self.station_manager: StationManager = None
        self.shape = 'empty-space'
        self.short_path = self.construct_station_path(data, x, y)
        self.long_path = self.construct_station_path(data, x, y, short_path=False)
        self.order_ids: List[int] = []
        self.orders: List[Order] = []
        self.max_orders = 6 # Picking station capacity
        self.short_path_threshold = 4
        self.robot_ids = {}
        self.is_using_short_route = True
        self.skus = {} # {A:15, B: 10}
        self.skus_in_station = {} # {A:[5,10], B:[10]}
        self.incoming_pod: List[int] = []

    def setStationManager(self, station_manager):
        self.station_manager = station_manager

    def addOrder(self, order_id: int, order:Order):
        self.order_ids.append(order_id)
        self.orders.append(order)

        for sku, value in order.getRemainingSKU().items():
            if sku not in self.skus_in_station:
                self.skus_in_station[sku] = []
            self.skus_in_station[sku].append(value)

    def reduceSKUFromStation(self, sku, value):
        if sku in self.skus_in_station and value in self.skus_in_station[sku]:
            self.skus_in_station[sku].remove(value)
            if len(self.skus_in_station[sku]) == 0:
                self.skus_in_station.pop(sku)

    def removeOrder(self, order_id: int, order: Order):
        if order_id in self.order_ids:
            self.order_ids.remove(order_id)
        if order in self.orders:
            self.orders.remove(order)

    def addPod(self, pod):
        self.incoming_pod.append(pod)
    
    def removePod(self, pod):
        self.incoming_pod.remove(pod)

    def isPickerStation(self) -> bool:
        return self.object_type == "picker"

    def isReplenishmentStation(self) -> bool:
        return self.object_type == "replenishment"

    def getPath(self):
        if self.is_using_short_route:
            return self.short_path
        else:
            return self.long_path

    def getSubPath(self, robot_id,x: int, y: int):
        sub_path = []
        start = False
        for coord in self.getRobotRoute(robot_id):
            if (coord.x, coord.y) == (x, y):
                start = True
            if start:
                sub_path.append(NetLogoCoordinate(coord.x, coord.y))
        return sub_path

    def reevaluateRoute(self):
        if self.is_using_short_route and len(self.robot_ids) > self.short_path_threshold:
            self.is_using_short_route = False
        elif not self.is_using_short_route and len(self.robot_ids) == 0:
            self.is_using_short_route = True

    def addRobot(self, robot_id):
        self.robot_ids[robot_id] = self.getPath()
        self.reevaluateRoute()

    def removeRobot(self, robot_id):
        if robot_id in self.robot_ids:
            del self.robot_ids[robot_id]
        self.reevaluateRoute()

    def updateRobotRouteType(self, robot_id):
        if robot_id in self.robot_ids:
            self.robot_ids[robot_id] = self.getPath()

    def getRobotRoute(self, robot_id):
        return self.robot_ids.get(robot_id, None)

    def hasRouteChanged(self, robot_id):
        return self.getPath() != self.getRobotRoute(robot_id)
    
    def getSKUsInStation(self):
        for sku, value in self.skus_in_station.items():
            self.skus[sku] = sum(value)
        return self.skus
    
    def getOrdersInStation(self) -> Optional[List[Order]]: 
        return self.orders
    
    def getSKUsInStationDict(self) -> Optional[Dict]:
        return self._sortOrderSet(self.skus_in_station)

    def _sortOrderSet(self,order_set):
        sorted_order_set = {}
        for index, value in order_set.items():
            sorted_order_set[index] = sorted(value, reverse=False)
        return sorted_order_set
    
    def construct_station_path(self, data: DataFrame, start_x, start_y, short_path=True):
        station_path: List[NetLogoCoordinate] = [NetLogoCoordinate(start_x, start_y)]

        x_increment = 1 if self.object_type == 'picker' else -1
        if not short_path:
            station_path.insert(0, NetLogoCoordinate(start_x + 1 * x_increment, start_y))
            station_path.insert(0, NetLogoCoordinate(start_x + 2 * x_increment, start_y))
            station_path.insert(0, NetLogoCoordinate(start_x + 2 * x_increment, start_y + 1))
            station_path.insert(0, NetLogoCoordinate(start_x + 1 * x_increment, start_y + 1))

        # go to bottom
        y, x = start_y + 1, start_x
        while data.iloc[y, x] in (14, 17, 24, 27):
            station_path.insert(0, NetLogoCoordinate(x, y))

            if data.iloc[y, x] in (17, 27):
                x += x_increment
                while data.iloc[y, x] in (13, 23):
                    station_path.insert(0, NetLogoCoordinate(x, y))
                    x += x_increment

            y += 1

        return station_path
