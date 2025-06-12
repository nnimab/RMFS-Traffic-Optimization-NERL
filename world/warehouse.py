from __future__ import annotations
import os
from typing import Optional, List, TYPE_CHECKING

import pandas as pd
from lib.types.directed_graph import DirectedGraph
from world.layout import Layout
from world.landscape import Landscape
from lib.math import *
from world.managers.intersection_manager import IntersectionManager
from world.managers.order_manager import OrderManager
from world.managers.zone_manager import ZoneManager
from world.managers.job_manager import JobManager
from world.managers.robot_manager import RobotManager
from world.managers.pod_manager import PodManager
from world.managers.area_path_manager import AreaPathManager
from world.managers.station_manager import StationManager
from world.entities.order import Order
from world.entities.pod import Pod
from world.entities.robot import Robot
from world.entities.job import Job
from lib.generator.order_generator import *
from lib.constant import *
if TYPE_CHECKING:
    from world.entities.object import Object

class Warehouse:
    DIMENSION = 60
    def __init__(self):
        self._tick = 0
        self.ignored_types = ["pod", "station", "area_path", "intersection"]
        self.job_queue = []
        self.stop_and_go = 0
        self.total_energy = 0
        self.total_pod = 0
        self.total_turning = 0
        self.warehouse_size = []
        self.layout = Layout()
        self.landscape = Landscape(self.DIMENSION)
        self.order_manager = OrderManager(self)
        self.zone_manager = ZoneManager(self)
        self.job_manager = JobManager(self)
        self.intersection_manager = IntersectionManager(self, self.landscape.current_date_string)
        self.area_path_manager = AreaPathManager(self)
        self.robot_manager = RobotManager(self)
        self.pod_manager = PodManager(self)
        self.station_manager = StationManager(self)
        self.next_process_tick = 0
        self.update_intersection_using_RL = True
        self.zoning = False
        self.graph = DirectedGraph()
        self.graph_pod = DirectedGraph()
        self.current_controller = "none"  # 存儲當前使用的控制器類型

    def initWarehouse(self):
        self.robot_manager.initRobotManager()
        self.station_manager.initStationManager()
        self.pod_manager.initPodManager()
        # area path and intersection entity don't need connection back to the managers
        # self.area_path_manager.initAreaPathManager()
        # self.intersection_manager.initIntersectionManager()

    def setWarehouseSize(self, size):
        self.warehouse_size = size

    def getWarehouseSize(self):
        return self.warehouse_size

    def getObjects(self):
        result = []
        result.extend(self.area_path_manager.getAllAreaPaths())
        result.extend(self.intersection_manager.getAllIntersections())
        result.extend(self.pod_manager.getAllPods())
        result.extend(self.robot_manager.getAllRobots())
        result.extend(self.station_manager.getAllStations())
        return result
    
    def getMovableObjects(self):
        result = []
        for o in self.getObjects():
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result

    def tick(self):
        if int(self._tick) == self.next_process_tick:
            self.findNewOrders()
            self.processOrders()
            if self.update_intersection_using_RL:
                self.intersection_manager.update_traffic_using_controller(int(self._tick))
        if len(self.job_queue) > 0:
            current_distance = 1000000
            nearest_robot: Optional[Robot] = None

            for o in self.getMovableObjects():
                if len(self.job_queue) > 0:
                    job: Job = self.job_queue[0]

                    if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                        dist = calculate_distance(o.pos_x, o.pos_y, job.pod_coordinate.x, job.pod_coordinate.y)
                        if dist < current_distance:
                            nearest_robot = o
                            current_distance = dist

            if nearest_robot is not None:
                job: Job = self.job_queue.pop(0)
                nearest_robot.assignJobAndSetToTakePod(job)

        total_energy = 0
        total_turning = 0
        for o in self.getMovableObjects():
            initial_velocity = o.velocity
            o.move()
            if isinstance(o, Robot):
                total_energy += o.energy_consumption
                total_turning += o.turning
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

                if o.job is not None and o.job.picking_delay == 0 and not o.job.is_finished:
                    need_replenish_pod = self.finishTaskInJob(o.job)
                    if need_replenish_pod:
                        print(f"cihuy masuk")
                        pod: Pod = self.pod_manager.getPodsByCoordinate(o.job.pod_coordinate.x, o.job.pod_coordinate.y)
                        station_replenish = self.station_manager.findAvailableReplenishmentStation()
                        # Check if a replenishment station was found
                        if station_replenish:
                            new_job = self.job_manager.createJob(pod.coordinate, station_id=station_replenish.id)
                            new_job.addReplenishmentTask(pod)
                            o.assignJobAndSetToStation(new_job)
                        else:
                            # Handle the case where no replenishment station is available
                            print(f"WARNING: No available replenishment station found for pod at {pod.coordinate}. Replenishment job not created.")
                            # Option: Decide if the robot should do something else, e.g., return pod to storage or wait
                            # For now, just letting the robot potentially become idle after this job
                            self.pod_manager.setPodAvailable(o.job.pod_coordinate) # Make the pod available again
                            o.job.is_finished = True # Mark the original job as finished to avoid reprocessing
                            o.job = None # Clear the robot's job

                if o.current_state == 'idle' and o.job is not None:
                    self.pod_manager.setPodAvailable(o.job.pod_coordinate)
                    o.job = None

        self.total_energy = total_energy
        self.total_turning = total_turning

        if int(self._tick) == self.next_process_tick:
            self.next_process_tick += 1
            if self.update_intersection_using_RL:
                self.intersection_manager.updateModelAfterExecution(self._tick)

        self._tick += TICK_TO_SECOND

    def finishTaskInJob(self, job: Job):
        job_station = self.station_manager.getStationById(job.station_id)
        if job_station.isPickerStation():
            return self.finishPickingTask(job)
        elif job_station.isReplenishmentStation():
            return self.finishReplenishmentTask(job)
    
    def finishPickingTask(self, job: Job):
        # 根據工作中的貨架座標獲取貨架對象
        pod: Pod = self.pod_manager.getPodsByCoordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        # 初始化需要補貨的SKU列表
        sku_need_replenished = []
        # 遍歷工作中的所有訂單項目
        for order_id, sku, quantity in job.orders:
            # 根據訂單ID獲取訂單對象
            order: Order = self.order_manager.getOrderById(order_id)
            # 更新訂單中已配送的商品數量
            order.deliverQuantity(sku, quantity)
            # 輸出訂單處理信息
            print("order, sku, quantity :" ,order_id, sku, quantity)  # 調試輸出：顯示當前處理的訂單、SKU和數量

            # 從貨架上取出指定數量的SKU
            pod.pickSKU(sku, quantity)  # 從貨架上減少相應SKU的數量

            # 檢查SKU是否需要補貨
            # sku是SKU的唯一標識符(字符串)
            # 減少全局SKU數據中的庫存量
            self.pod_manager.reduceSKUData(sku, quantity)  # 更新全局SKU庫存數據
            # 檢查SKU是否需要補貨
            sku, replenished_status = self.pod_manager.isSKUNeedReplenishment(sku)  # 檢查SKU是否達到補貨閾值

            # 如果SKU需要補貨，將其添加到需補貨列表
            if(replenished_status == True): sku_need_replenished.append(sku)  # 將需要補貨的SKU加入列表
    
            # 更新訂單分配狀態
            file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"  # 訂單分配文件路徑
            assign_order_df = pd.read_csv(file_path)  # 讀取訂單分配數據
            # 將對應訂單項的狀態更新為已完成(1)
            assign_order_df.loc[((assign_order_df['order_id'] == order.id) & (assign_order_df['item_id'] == sku)), 'status'] = 1  # 更新狀態為已完成
            assign_order_df.to_csv(file_path, index=False)  # 保存更新後的訂單分配數據
            
            # 檢查訂單是否已全部完成
            if order.isOrderCompleted():  # 如果訂單中所有項目都已配送完成
                # 完成訂單並記錄完成時間
                self.order_manager.finishOrder(order_id, int(self._tick))  # 更新訂單完成狀態和時間
                # 從工作站移除已完成的訂單
                station = self.station_manager.getStationById(order.station_id)  # 獲取訂單所在工作站
                station.removeOrder(order_id,order)  # 從工作站移除已完成訂單
                # 將已完成的訂單信息寫入CSV文件
                self.insertFinishedOrderToCSV(order)  # 記錄已完成訂單的詳細信息

        # # TRACY
        # # Get pod that have SKU that need to be replenished
        # unique_sku_need_replenished = list(set(sku_need_replenished))
        # replenished_pod_needed_by_sku = self.pod_manager.getPodNeedReplenishment(unique_sku_need_replenished)
        # # Determine which pod will be Replenished
        # pod_id_will_be_replenished = self.pod_manager.determinePodWillBeReplenished(replenished_pod_needed_by_sku)
        # # Get the pod that will be Replenished
        # pod_will_be_replenished = self.pod_manager.getPodByNumber(pod_id_will_be_replenished)

        # 完成任務並檢查是否需要補貨
        job.is_finished = True
        
        # 如果有任何SKU需要補貨就返回True
        if len(sku_need_replenished) > 0:
            return True
            
        # 檢查整個貨架是否需要補貨
        need_replenish_pod = pod.isNeedReplenishment()
        print(f"Pod replenishment status: {need_replenish_pod}")
        return need_replenish_pod
    
    def finishReplenishmentTask(self, job: Job):
        pod: Pod = self.pod_manager.getPodsByCoordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod.replenishAllSKU()
        job.is_finished = True
        return False

    def insertFinishedOrderToCSV(self, order: Order):
        header = ["order_id", "order_arrival", "process_start_time", "order_complete_time", "station_id"]
        data = [order.id, order.order_arrival, order.process_start_time, order.order_complete_time,
                order.station_id]

        write_to_csv("order-finished.csv", header, data, self.landscape.current_date_string)

    def findNewOrders(self):
        order_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_order.csv')
        orders_df = pd.read_csv(order_path)

        file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        if os.path.exists(file_path):
            assign_order_df = pd.read_csv(file_path)
            # pass
        else:
            assign_order_df = orders_df.copy()
            assign_order_df['assigned_station'] = None
            assign_order_df['assigned_pod'] = None
            assign_order_df['status'] = -3
            assign_order_df.to_csv(file_path, index=False)
        new_file_df = pd.read_csv(file_path)
                  
        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = new_file_df[(new_file_df['order_arrival']<= current_second) & 
                               (new_file_df['order_arrival'] > previous_second) &
                               (new_file_df['status'] == -3)]
        grouped_orders = new_orders.groupby('order_id')

        for order_id, group in grouped_orders:
            order_items = group[['item_id', 'item_quantity']].to_dict('records')
            order = self.order_manager.createOrder(order_id, current_second)

            # Add each item in the group to the order
            for item in order_items:
                order.addSKU(item['item_id'], item['item_quantity'])

        return new_orders

    def assignJobToAvailableRobot(self, job: Job):
        current_distance = 1000000
        current_id = -1

        for o in self.getMovableObjects():
            if isinstance(o, Robot) and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                dist = calculate_distance(o.pos_x, o.pos_y, job.pod_coordinate.x, job.pod_coordinate.y)
                if dist < current_distance:
                    current_id = o.id
                    current_distance = dist

        if current_id == -1:
            self.job_queue.append(job)
            return

        for o in self.getMovableObjects():
            if o.id == current_id and isinstance(o, Robot):
                o.assignJobAndSetToTakePod(job)

    def processOrders(self):
        file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        robots_location = []
        for o in self.getMovableObjects():
            if len(self.job_queue) > 0:
                job: Job = self.job_queue[0]

                if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                    robots_location.append([o.pos_x, o.pos_y])

        for order in self.order_manager.unfinished_orders:
            assign_order_df = pd.read_csv(file_path)
            if order.station_id is None:
                # available_station = self.station_manager.findAvailablePickingStation()
                available_station = self.station_manager.findHighestSimilarityStation(order.skus, self.pod_manager)
                if available_station is not None:
                    order.assignStation(available_station.id)
                    available_station.addOrder(order.id, order)

                    assign_order_df.loc[assign_order_df['order_id'] == order.id, 'assigned_station'] = available_station.id
                    assign_order_df.loc[assign_order_df['order_id'] == order.id, 'status'] = -1
                else:
                    break

            if order.process_start_time <= 0:
                order.startProcessing(int(self._tick))

            assign_order_df.to_csv(file_path, index=False)

            
            # Get the station assigned to this order and orders in that station
            order_station = self.station_manager.getStationById(order.station_id)
            orders_in_station = order_station.getOrdersInStation()

            # For Emily {A:10, B:5, C:12}
            skus_in_station = order_station.getSKUsInStation()

            # For Jhen {A:[5,5], B:[5], C:[3,4,5]}
            skus_in_station_dict = order_station.getSKUsInStationDict()
            
            station_coordinate = order_station.coordinate
            for sku in order.getRemainingSKU():
                 # This is the baseline
                # available_pod: Pod = self.pod_manager.getAvailablePod(sku) 
                
                # This is Emily's pod picking
                available_pod: Pod = self.pod_manager.getAvailablePodSimilarity(sku, skus_in_station, station_coordinate, robots_location) 
                # This is Jhen's pod picking
                # available_pod: Pod = self.pod_manager.getAvailablePodInventory(sku, skus_in_station_dict, station_coordinate, robots_location) 
                if available_pod is None:
                    continue
                quantity_to_take = order.getQuantityLeftForSKU(sku)
                order.commitQuantity(sku, quantity_to_take)

                # Commiting every order that has the sku in the pod chosen
                available_pod.pickSKU(sku, quantity_to_take)
                
                 # Append pod to station
                order_station.addPod(available_pod.pod_number)
                available_pod.station = order_station

                assign_order_df.loc[((assign_order_df['order_id'] == order.id) & (assign_order_df['item_id'] == sku)), 'assigned_pod'] = int(available_pod.pod_number)
                
                assign_order_df.loc[((assign_order_df['order_id'] == order.id) & (assign_order_df['item_id'] == sku)), 'status'] = 0

                assign_order_df.to_csv(file_path, index=False)

                job = self.job_manager.createJob(available_pod.coordinate, station_id=order.station_id)
                self.pod_manager.setPodNotAvailable(available_pod.coordinate)
                # print(f"sku {sku} quantity {quantity_to_take}")

                job.addPickingTask(order.id, sku, quantity_to_take) # Simple kan disini ya beb
                pod_skus = [i for i in available_pod.skus]
                
                # Turn this off for baseline 
                for skus_pod in pod_skus:
                    for order_ in orders_in_station:
                        if order_ != order and order_.hasSKU(skus_pod):
                                quantity_to_take_other = order_.getQuantityLeftForSKU(skus_pod)
                                if available_pod.getQuantity(skus_pod) > quantity_to_take_other and quantity_to_take_other > 0:
                                    order_.commitQuantity(skus_pod, quantity_to_take_other)
                                    available_pod.pickSKU(sku, quantity_to_take_other)
                                    job.addPickingTask(order_.id, skus_pod,quantity_to_take_other)

                for order_ in orders_in_station:
                    if order_ != order and order_.hasSKU(sku):
                            quantity_to_take_other = order.getQuantityLeftForSKU(sku)
                            if available_pod.getQuantity(sku) > quantity_to_take_other and quantity_to_take > 0:
                                # print(f"sku {sku} quantity {quantity_to_take}")
                                job.addPickingTask(order_.id, sku,quantity_to_take_other)

                self.job_queue.append(job)

    def generateResult(self):
        result = []
        for o in self.getMovableObjects():
            result.append({
                'id': o.id,
                'heading': o.heading,
                'shape': o.shape,
                'velocity': o.velocity,
                'acceleration': o.acceleration,
                'pos_x': o.pos_x,
                'pos_y': o.pos_y,
                'color': o.color,
            })

        return result

    def set_traffic_controller(self, controller_type, **kwargs):
        """
        設置交通控制器類型
        
        Args:
            controller_type (str): 控制器類型，例如 "time_based", "queue_based", "dqn", "nerl"
            **kwargs: 控制器需要的額外參數
            
        Returns:
            bool: 成功返回True，失敗返回False
        """
        success = self.intersection_manager.set_controller(controller_type, **kwargs)
        if success:
            # 啟用交通控制
            self.update_intersection_using_RL = True
            print(f"交通控制器已設置為: {controller_type}")
            self.current_controller = controller_type
        return success