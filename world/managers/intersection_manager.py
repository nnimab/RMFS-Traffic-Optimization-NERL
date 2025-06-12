from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Dict
from ai.deep_q_network import DeepQNetwork
from ai.traffic_controller import TrafficController, TrafficControllerFactory
from ai.controllers.nerl_controller import NEController
from lib.file import *
from world.entities.intersection import Intersection
if TYPE_CHECKING:
    from world.warehouse import Warehouse

class IntersectionManager:
    def __init__(self, warehouse: Warehouse, start_date_string):
        self.warehouse = warehouse
        self.intersection_counter = 0
        self.intersections: List[Intersection] = []
        self.coordinate_to_intersection = {}
        self.intersection_id_to_intersection = {}
        self.q_models = {}
        self.previous_state = {}
        self.previous_action = {}
        self.start_date_string = start_date_string
        self.controllers: Dict[str, TrafficController] = {}
        self.current_controller_type = None
        self.intersection_controllers = {}

    def initIntersectionManager(self):
        for intersection in self.intersections:
            intersection.setIntersectionManager(self)

    def getAllIntersections(self):
        return self.intersections
    
    def createIntersection(self, x: int, y: int):
        intersection = Intersection(self.intersection_counter, x, y)
        self.intersections.append(intersection)
        coordinate = intersection.coordinate
        self.intersection_counter += 1
        self.coordinate_to_intersection[(coordinate.x, coordinate.y)] = intersection
        self.intersection_id_to_intersection[intersection.id] = intersection
        return intersection

    def set_controller(self, controller_type, **kwargs):
        if controller_type not in self.controllers:
            try:
                self.controllers[controller_type] = TrafficControllerFactory.create_controller(controller_type, **kwargs)
            except (ImportError, ValueError) as e:
                print(f"Failed to create controller {controller_type}: {e}")
                return False
        
        self.current_controller_type = controller_type
        return True
    
    def update_traffic_using_controller(self, tick):
        if self.current_controller_type is None or self.current_controller_type not in self.controllers:
            print("No valid traffic controller set")
            return
        
        controller = self.controllers[self.current_controller_type]
        
        for intersection in self.intersections:
            # 獲取控制器決定的方向
            direction = controller.get_direction(intersection, tick, self.warehouse)
            
            # 更新方向如果需要
            if direction != intersection.allowed_direction:
                # 對於主要交叉路口(15,15)保留特殊的日誌輸出
                if intersection.pos_x == 15 and intersection.pos_y == 15:
                    print(f"Main intersection (15,15) direction change: {intersection.allowed_direction} -> {direction}")
                else:
                    print(f"Intersection {intersection.id} at ({intersection.pos_x}, {intersection.pos_y}) direction change: {intersection.allowed_direction} -> {direction}")
                self.updateAllowedDirection(intersection.id, direction, tick)
            
            # 如果是DQN控制器，對其進行訓練
            if self.current_controller_type == "dqn":
                try:
                    # 確保控制器有train方法
                    if hasattr(controller, 'train') and callable(controller.train):
                        controller.train(intersection, tick, self.warehouse)
                except Exception as e:
                    print(f"Error during DQN training: {e}")
            
            # 如果是NERL控制器，對其進行訓練
            if self.current_controller_type == "nerl":
                try:
                    # 確保控制器有train方法
                    if hasattr(controller, 'train') and callable(controller.train):
                        controller.train(intersection, tick, self.warehouse)
                except Exception as e:
                    print(f"Error during NERL training: {e}")
                    
        # --- AFTER THE LOOP ---
        # print(f"DEBUG: Finished intersection loop for tick {tick}. Preparing for evolution check.")
        # 在處理完所有交叉口後，檢查NERL進化條件
        if self.current_controller_type == "nerl" and isinstance(controller, NEController) and controller.is_training:
            # print(f"DEBUG: Triggering evolution check for tick {tick} AFTER intersection loop.")
            try:
                controller.step_evolution_counter_and_evolve(tick)
            except Exception as e:
                print(f"Error during NERL evolution step: {e}")
                    
    def getIntersectionByCoordinate(self, x, y):
        return self.coordinate_to_intersection.get((x, y), None)

    def getConnectedIntersections(self, current_intersection: Intersection) -> List[Intersection]:
        connected_intersections = []
        connected_intersection_ids = current_intersection.connected_intersection_ids

        for intersection_id in connected_intersection_ids:
            intersection = self.findIntersectionById(intersection_id)
            if intersection is not None:
                connected_intersections.append(intersection)

        return connected_intersections

    def getState(self, current_intersection: Intersection, tick):
        state = [
            current_intersection.getAllowedDirectionCode(),
            current_intersection.durationSinceLastChange(tick),
            len(current_intersection.horizontal_robots),
            len(current_intersection.getRobotsByStateHorizontal("delivering_pod")),
            len(current_intersection.getRobotsByStateHorizontal("returning_pod")),
            len(current_intersection.getRobotsByStateHorizontal("taking_pod")),
            len(current_intersection.vertical_robots),
            len(current_intersection.getRobotsByStateVertical("delivering_pod")),
            len(current_intersection.getRobotsByStateVertical("returning_pod")),
            len(current_intersection.getRobotsByStateVertical("taking_pod")),
        ]
        connected_intersections = self.getConnectedIntersections(current_intersection)
        for intersection in connected_intersections:
            state.append(intersection.getAllowedDirectionCode())
            state.append(intersection.robotCount())
        return state

    def handleModel(self, intersection: Intersection, tick):
        state = self.getState(intersection, tick)
        self.previous_state[intersection.id] = state
        if intersection.RL_model_name not in self.q_models:
            self.q_models[intersection.RL_model_name] = self.createNewModel(intersection, state)
        model = self.q_models[intersection.RL_model_name]
        action = model.act(state)

        self.intersectionToCsv(intersection, action, tick)

        self.previous_action[intersection.id] = action
        new_direction = intersection.getAllowedDirectionByCode(action)
        self.updateAllowedDirection(intersection.id, new_direction, tick)

    def intersectionToCsv(self, intersection, action, tick):
        previous_allowed_direction = intersection.allowed_direction
        new_allowed_direction = intersection.getAllowedDirectionByCode(action)
        if previous_allowed_direction == new_allowed_direction:
            return

        previous_allowed_direction = previous_allowed_direction if previous_allowed_direction is not None else "None"
        new_allowed_direction = new_allowed_direction if new_allowed_direction is not None else "None"

        header = ["intersection_id", "previous_action", "action_decided", "tick_changed", "durationSinceLastChange"]
        data = [
            intersection.id,
            previous_allowed_direction,
            new_allowed_direction,
            tick,
            intersection.durationSinceLastChange(tick)
        ]

        write_to_csv("allowed_direction_changes.csv", header, data, self.start_date_string)

    @staticmethod
    def createNewModel(intersection: Intersection, state):
        state_size = len(state)
        return DeepQNetwork(state_size=state_size,
                            action_size=3,
                            model_name=intersection.RL_model_name)

    def updateDirectionUsingDQN(self, tick):
        if self.current_controller_type == "dqn":
            self.update_traffic_using_controller(tick)
        else:
            # 如果不是DQN控制器，保持原代碼邏輯
            pass

    def updateModelAfterExecution(self, tick):
        pass

    def rememberAndReplay(self, intersection: Intersection, reward, done, tick):
        model = self.q_models[intersection.RL_model_name]
        if intersection.id in self.previous_state and intersection.id in self.previous_action:
            next_state = self.getState(intersection, tick)
            model.remember(self.previous_state[intersection.id],
                           self.previous_action[intersection.id], reward, next_state, done)
            if done:
                model.replay(64)

            self.resetStateAction(intersection)

        if tick % 1000 == 0 and tick != 0:
            print("SAVING_MODEL")
            intersection.resetTotals()
            model.save_model(intersection.RL_model_name, tick)

    def resetStateAction(self, intersection: Intersection):
        if intersection.RL_model_name in self.previous_state:
            del self.previous_state[intersection.id]
        if intersection.RL_model_name in self.previous_action:
            del self.previous_action[intersection.id]

    @staticmethod
    def isEpisodeDone(intersection: Intersection, tick):
        if intersection.robotCount() == 0:
            return True
        elif int(tick) % 1000 == 0:
            return True
        else:
            return False

    def calculateReward(self, intersection: Intersection, tick):
        reward = 0

        for each_robot in intersection.previous_vertical_robots:
            reward += self.calculatePassingRobotReward(each_robot, intersection, "vertical", 2)

        for each_robot in intersection.previous_horizontal_robots:
            reward += self.calculatePassingRobotReward(each_robot, intersection, "horizontal", 1)

        intersection.clearPreviousRobots()

        for each_robot in intersection.vertical_robots.values():
            reward += self.calculateCurrentRobotReward(each_robot, intersection, "vertical", 2, tick)

        for each_robot in intersection.horizontal_robots.values():
            reward += self.calculateCurrentRobotReward(each_robot, intersection, "horizontal", 1, tick)

        if intersection.allowed_direction is not None and intersection.robotCount() == 0:
            reward += -0.1

        return reward

    def calculateCurrentRobotReward(self, robot, intersection, direction, multiplier, current_tick):
        robot_state_multiplier = self.getStateMultiplier(robot)

        total_waiting_time_current_robot = current_tick - robot.current_intersection_start_time
        average_waiting_time = intersection.calculateAverageWaitingTime(direction)

        total_stop_n_go_current_robot = robot.current_intersection_stop_and_go
        average_stop_n_go = intersection.calculateAverageStopAndGo(direction)

        reward = 0
        if total_waiting_time_current_robot > average_waiting_time:
            wait_diff = total_waiting_time_current_robot - average_waiting_time
            reward += -0.1 * wait_diff * robot_state_multiplier * multiplier

        if total_stop_n_go_current_robot > average_stop_n_go:
            stop_go_diff = total_stop_n_go_current_robot - average_stop_n_go
            reward += -0.1 * stop_go_diff * robot_state_multiplier * multiplier

        return reward

    def calculatePassingRobotReward(self, robot, intersection, direction, multiplier):
        robot_state_multiplier = self.getStateMultiplier(robot)

        previous_average_wait = intersection.calculateAverageWaitingTime(direction)
        previous_average_stop_n_go = intersection.calculateAverageStopAndGo(direction)

        intersection.trackRobotIntersectionData(robot, direction)

        current_average_wait = intersection.calculateAverageWaitingTime(direction)
        current_average_stop_n_go = intersection.calculateAverageStopAndGo(direction)

        reward = 0
        if current_average_wait < previous_average_wait:
            wait_diff = previous_average_wait - current_average_wait
            reward += 0.3 * wait_diff * robot_state_multiplier * multiplier

        if current_average_stop_n_go < previous_average_stop_n_go:
            stop_go_diff = previous_average_stop_n_go - current_average_stop_n_go
            reward += 0.3 * stop_go_diff * robot_state_multiplier * multiplier

        reward += 1 * robot_state_multiplier * multiplier

        return reward

    @staticmethod
    def getStateMultiplier(robot):
        if robot.current_state == 'delivering_pod':
            return 1.5
        elif robot.current_state == 'returning_pod':
            return 1
        elif robot.current_state == 'taking_pod':
            return 0.75
        else:
            return 1

    def updateAllowedDirection(self, intersection_id, direction, tick):
        intersection: Intersection = self.findIntersectionById(intersection_id)
        intersection.changeTrafficLight(direction, tick)

    def findIntersectionByCoordinate(self, x: int, y: int) -> Optional[str]:
        for intersection in self.intersections:
            # 檢查坐標是否在接近路徑坐標列表中
            if (x, y) in intersection.approaching_path_coordinates:
                return intersection.id
            
            # 如果不在接近路徑中，檢查是否在交叉路口的近距離範圍內
            intersection_x = intersection.coordinate.x
            intersection_y = intersection.coordinate.y
            distance = abs(x - intersection_x) + abs(y - intersection_y)  # 曼哈頓距離
            
            # 如果交叉路口是主要交叉路口(15,15)，使用更大的識別半徑(3個單位)
            if intersection_x == 15 and intersection_y == 15:
                if distance <= 3:
                    print(f"Main intersection detection: Robot at ({x}, {y}) is near main intersection ({intersection_x}, {intersection_y}), distance: {distance}")
                    return intersection.id
            # 一般交叉路口使用2個單位的識別半徑
            elif distance <= 2:
                print(f"Intersection detection: Robot at ({x}, {y}) is near intersection ({intersection_x}, {intersection_y}), ID: {intersection.id}")
                return intersection.id
                    
        return None

    def findIntersectionById(self, intersection_id):
        return self.intersection_id_to_intersection.get(intersection_id, None)

    def printInfo(self, x, y):
        intersection = self.coordinate_to_intersection.get((x, y))
        if intersection is not None:
            intersection.printInfo()
