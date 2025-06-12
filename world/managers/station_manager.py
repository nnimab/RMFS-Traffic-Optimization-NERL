from __future__ import annotations
from typing import List, Optional, Dict, TYPE_CHECKING
from world.entities.station import Station
from world.entities.picker import Picker
from world.entities.replenishment import Replenishment
from .pod_manager import PodManager
import pandas as pd
import numpy as np
if TYPE_CHECKING:
    from world.warehouse import Warehouse

class StationManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.picker_counter = 0
        self.picking_stations: List[Station] = []
        self.replenishment_counter = 0
        self.replenishment_stations: List[Station] = []
        self.stations_by_id: Dict[int, Station] = {}

    def initStationManager(self):
        for station in self.getAllStations():
            station.setStationManager(self)

    def getAllStations(self):
        return self.picking_stations + self.replenishment_stations
    
    def getStationById(self, station_id):
        return self.stations_by_id[station_id]
    
    def addStation(self, station: Station):
        self.stations_by_id[station.id] = station

        if station.isPickerStation():
            self.picking_stations.append(station)
        elif station.isReplenishmentStation():
            self.replenishment_stations.append(station)

    def createPickerStation(self, x: int, y: int, data: pd.DataFrame):
        obj = Picker(self.picker_counter, x, y, data)
        self.picker_counter += 1
        self.picking_stations.append(obj)
        self.stations_by_id[obj.id] = obj
    
    def createReplenishmentStation(self, x: int, y: int, data: pd.DataFrame, max_robots: int = 3):
        obj = Replenishment(self.replenishment_counter, x, y, data, max_robots)
        self.replenishment_counter += 1
        self.replenishment_stations.append(obj)
        self.stations_by_id[obj.id] = obj
    
    def findAvailablePickingStation(self) -> Optional[Station]:
        # Initialize the available station variable as None
        available_station = None
        # Initialize the minimum number of orders to a high value to find the station with the least orders
        min_orders = float('inf')

        # Iterate through each station to check the number of orders
        for station in self.picking_stations:
            if len(station.order_ids) < station.max_orders:
                # Check if this station has fewer orders than the current minimum
                if len(station.order_ids) < min_orders:
                    min_orders = len(station.order_ids)
                    available_station = station

        return available_station
    
    def findAvailableReplenishmentStation(self) -> Optional[Station]:
        """
        尋找可用的補貨站。
        優先選擇有空位且機器人數量最少的站點。
        如果所有站點都滿了，則選擇當前機器人數量最少的站點（即使已滿）。
        """
        available_station: Optional[Station] = None
        min_robots_below_capacity = float('inf')
        
        # 首先，尋找有空位的站點
        for station in self.replenishment_stations:
            if len(station.robot_ids) < station.max_robots:
                if len(station.robot_ids) < min_robots_below_capacity:
                    min_robots_below_capacity = len(station.robot_ids)
                    available_station = station
                    
        # 如果找到了有空位的站點，直接返回
        if available_station:
            # print(f"DEBUG: Found available replenishment station {available_station.id} with {min_robots_below_capacity}/{available_station.max_robots} robots.")
            return available_station
            
        # 如果沒有找到有空位的站點 (所有站點都滿了)，則從所有站點中選擇機器人最少的那個
        # (前提是至少有一個補貨站存在)
        if not self.replenishment_stations:
            print("WARNING: No replenishment stations exist in the system.")
            return None
            
        min_robots_overall = float('inf')
        fallback_station: Optional[Station] = None
        for station in self.replenishment_stations:
            if len(station.robot_ids) < min_robots_overall:
                min_robots_overall = len(station.robot_ids)
                fallback_station = station
                
        if fallback_station:
            print(f"WARNING: All replenishment stations are at capacity ({fallback_station.max_robots}). Assigning to station {fallback_station.id} with {min_robots_overall} robots anyway.")
        else:
             # This case should theoretically not happen if replenishment_stations is not empty
             print("ERROR: Could not select a fallback replenishment station.")
             
        return fallback_station

    def findHighestSimilarityStation(self, skus_in_order, pod_manager: PodManager) -> Optional[Station]:
        available_station_rank = pd.DataFrame(columns=["station_id", "priority_score"])
        sku_in_order_list = [i for i in skus_in_order]
        available_station = []
        assignStation = None

        # Store all available station
        for station in self.picking_stations:
            if len(station.order_ids) < station.max_orders:
                available_station.append(station)
        
        # Check if more than one station is available
        if len(available_station) > 1:
            for station in available_station:
                # Take pod assigned to this particular station
                station_incoming_pod = station.incoming_pod
                station_pod_skus_set = set()
                for pod_id in station_incoming_pod:
                    pod = pod_manager.getPodByNumber(pod_id)
                    if pod:
                        pod_skus = [item for item, details in pod.skus.items() if details['current_qty'] > 0]
                        station_pod_skus_set.update(pod_skus)

                station_pod_skus_list = list(station_pod_skus_set)
                station_pod_skus_in_order_mask = np.isin(sku_in_order_list, station_pod_skus_list)
                station_pod_skus_in_order = np.array(sku_in_order_list)[station_pod_skus_in_order_mask]
                similarity_score = len(station_pod_skus_in_order)

                # Calculate load score (fraction of available capacity, higher is better)
                if station.max_orders > 0:
                    load_score = (station.max_orders - len(station.order_ids)) / station.max_orders
                else:
                    load_score = 0 # Should not happen for pickers, but safe check

                # Combine scores (equal weight for now)
                # priority_score = similarity_score + load_score # Old weighting
                # New weighting: Scale load_score by max_orders to make it more comparable to similarity_score
                priority_score = similarity_score + (station.max_orders * load_score)

                available_station_rank = pd.concat([available_station_rank ,
                                            pd.DataFrame([[station.id, priority_score]], columns=["station_id", "priority_score"])], ignore_index=True)
        
            available_station_rank.sort_values(by=["priority_score"], ascending=False, inplace=True)
            available_station_rank.reset_index(drop=True, inplace=True)

            if len(available_station_rank) > 0:
                assignStation_id = available_station_rank.loc[0, "station_id"]
                assignStation = self.getStationById(assignStation_id)
        elif len(available_station) == 1:
            assignStation = available_station[0]

        return assignStation

    
