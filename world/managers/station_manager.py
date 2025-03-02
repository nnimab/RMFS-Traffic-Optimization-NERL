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
    
    def createReplenishmentStation(self, x: int, y: int, data: pd.DataFrame):
        obj = Replenishment(self.replenishment_counter, x, y, data)
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
        # Initialize the available station variable as None
        available_station = None
        # Initialize the minimum number of orders to a high value to find the station with the least orders
        min_orders = float('inf')

        # Iterate through each station to check the number of orders
        for station in self.replenishment_stations:
            if len(station.order_ids) < station.max_orders:
                # Check if this station has fewer orders than the current minimum
                if len(station.order_ids) < min_orders:
                    min_orders = len(station.order_ids)
                    available_station = station

        return available_station

    def findHighestSimilarityStation(self, skus_in_order, pod_manager: PodManager) -> Optional[Station]:
        available_station_rank = pd.DataFrame(columns=["station_id", "similarity_score"])
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
                    pod  = pod_manager.getPodByNumber(pod_id)
                    pod_skus = [item for item, details in pod.skus.items() if details['current_qty'] > 0]
                    station_pod_skus_set.update(pod_skus)

                station_pod_skus_list = list(station_pod_skus_set)
                station_pod_skus_in_order_mask = np.isin(sku_in_order_list, station_pod_skus_list)
                station_pod_skus_in_order = np.array(sku_in_order_list)[station_pod_skus_in_order_mask]
                similarity_score = len(station_pod_skus_in_order)

                available_station_rank = pd.concat([available_station_rank , 
                                            pd.DataFrame([[station.id, similarity_score]], columns=["station_id", "similarity_score"])], ignore_index=True) 
        
            available_station_rank.sort_values(by=["similarity_score"], ascending=False, inplace=True)
            available_station_rank.reset_index(drop=True, inplace=True)

            if len(available_station_rank) > 0:
                assignStation_id = available_station_rank.loc[0, "station_id"]
                assignStation = self.getStationById(assignStation_id)
        elif len(available_station) == 1:
            assignStation = available_station[0]

        return assignStation

    
