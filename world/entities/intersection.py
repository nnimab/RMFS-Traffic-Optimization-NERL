from typing import TYPE_CHECKING
from world.entities.object import Object
if TYPE_CHECKING:
    from world.managers.intersection_manager import IntersectionManager

class Intersection(Object):
    def __init__(self, id: int, x: int, y: int, use_reinforcement_learning=False):
        super().__init__(id, 'intersection', x, y)
        self.intersection_manager: IntersectionManager = None
        self.approaching_path_coordinates = [(x, y)]
        self.allowed_direction = None
        self.vertical_robots = {}
        self.horizontal_robots = {}
        self.previous_vertical_robots = []
        self.previous_horizontal_robots = []
        self.last_changed_tick = 0
        self.use_reinforcement_learning = use_reinforcement_learning
        self.RL_model_name = None
        self.connected_intersection_ids = []
        self.total_stop_n_go_horizontal = 0
        self.total_stop_n_go_vertical = 0
        self.total_waiting_time_horizontal = 0
        self.total_waiting_time_vertical = 0
        self.total_robots_passed_horizontal = 0
        self.total_robots_passed_vertical = 0
        
        # 新增屬性：用於計算平均交叉口流量
        self.total_robots_passed = 0  # 已通過的機器人總數
        self.simulation_start_tick = 0  # 模擬開始的tick
        self.robot_pass_records = []  # 記錄每個機器人通過的時間，用於計算單位時間的流量
        
        # 新增屬性：用於更精確計算平均等待時間
        self.total_waiting_time = 0  # 所有機器人在交叉口等待的總時間
        self.waiting_time_records = []  # 記錄每個機器人的等待時間

    def setIntersectionManager(self, intersection_manager):
        self.intersection_manager = intersection_manager

    def durationSinceLastChange(self, tick):
        return tick - self.last_changed_tick

    def addRobot(self, robot):
        # Consider precision issues, use more tolerant comparison
        x_diff = abs(robot.pos_x - self.coordinate.x)
        y_diff = abs(robot.pos_y - self.coordinate.y)
        
        # Special debug output for main intersection (15,15)
        is_main_intersection = self.coordinate.x == 15 and self.coordinate.y == 15
        if is_main_intersection:
            print(f"Attempting to add robot to main intersection: Robot position({robot.pos_x}, {robot.pos_y}), Intersection position({self.coordinate.x}, {self.coordinate.y})")
            print(f"X difference: {x_diff}, Y difference: {y_diff}")
        
        # Vertical direction: robot's x coordinate is close to intersection's x coordinate, and y difference is larger
        if x_diff < 0.5:  # Allow some error margin
            if is_main_intersection:
                print(f"Adding vertical direction robot: {robot.robotName()}")
            self.vertical_robots[robot.robotName()] = robot
        # Horizontal direction: robot's y coordinate is close to intersection's y coordinate, and x difference is larger
        elif y_diff < 0.5:  # Allow some error margin
            if is_main_intersection:
                print(f"Adding horizontal direction robot: {robot.robotName()}")
            self.horizontal_robots[robot.robotName()] = robot
        # If near the intersection but not in horizontal or vertical direction, decide based on robot's heading
        elif x_diff <= 3 and y_diff <= 3:  # Within 3 units of the intersection
            # Robot heading 0 or 180 is considered vertical direction
            if robot.heading == 0 or robot.heading == 180:
                if is_main_intersection:
                    print(f"Adding vertical direction robot based on heading: {robot.robotName()}, Heading: {robot.heading}")
                self.vertical_robots[robot.robotName()] = robot
            # Robot heading 90 or 270 is considered horizontal direction
            elif robot.heading == 90 or robot.heading == 270:
                if is_main_intersection:
                    print(f"Adding horizontal direction robot based on heading: {robot.robotName()}, Heading: {robot.heading}")
                self.horizontal_robots[robot.robotName()] = robot

    def removeRobot(self, robot):
        if robot.robotName() in self.horizontal_robots:
            del self.horizontal_robots[robot.robotName()]
            self.previous_horizontal_robots.append(robot)
        elif robot.robotName() in self.vertical_robots:
            del self.vertical_robots[robot.robotName()]
            self.previous_vertical_robots.append(robot)

    def getRobotsByStateHorizontal(self, state):
        return [robot for robot in self.horizontal_robots.values() if robot.current_state == state]

    def getRobotsByStateVertical(self, state):
        return [robot for robot in self.vertical_robots.values() if robot.current_state == state]

    def clearPreviousRobots(self):
        self.previous_horizontal_robots.clear()
        self.previous_vertical_robots.clear()

    def trackRobotIntersectionData(self, robot, direction):
        waiting_time = robot.current_intersection_finish_time - robot.current_intersection_start_time
        if direction == 'horizontal':
            self.total_stop_n_go_horizontal += robot.current_intersection_stop_and_go
            self.total_waiting_time_horizontal += waiting_time
            self.total_robots_passed_horizontal += 1
        elif direction == 'vertical':
            self.total_stop_n_go_vertical += robot.current_intersection_stop_and_go
            self.total_waiting_time_vertical += waiting_time
            self.total_robots_passed_vertical += 1

    def calculateAverageStopAndGo(self, direction):
        if direction == 'horizontal':
            return int(self.total_stop_n_go_horizontal / self.total_robots_passed_horizontal) \
                if self.total_robots_passed_horizontal > 0 else 0
        elif direction == 'vertical':
            return int(self.total_stop_n_go_vertical / self.total_robots_passed_vertical) \
                if self.total_robots_passed_vertical > 0 else 0

    def calculateAverageWaitingTime(self, direction):
        if direction == 'horizontal':
            return int(self.total_waiting_time_horizontal / self.total_robots_passed_horizontal) \
                if self.total_robots_passed_horizontal > 0 else 0
        elif direction == 'vertical':
            return int(self.total_waiting_time_vertical / self.total_robots_passed_vertical) \
                if self.total_robots_passed_vertical > 0 else 0

    def calculateTotalWaitingTimeCurrentRobots(self, direction, tick):
        total_waiting_time = 0

        if direction == 'horizontal':
            for robot in self.horizontal_robots.values():
                if robot.current_intersection_start_time is not None:
                    total_waiting_time += tick - robot.current_intersection_start_time
        elif direction == 'vertical':
            for robot in self.vertical_robots.values():
                if robot.current_intersection_start_time is not None:
                    total_waiting_time += tick - robot.current_intersection_start_time

        return total_waiting_time

    def resetTotals(self):
        self.total_stop_n_go_horizontal = 0
        self.total_stop_n_go_vertical = 0
        self.total_waiting_time_horizontal = 0
        self.total_waiting_time_vertical = 0
        self.total_robots_passed_horizontal = 0
        self.total_robots_passed_vertical = 0

    def robotCount(self):
        return len(self.horizontal_robots) + len(self.vertical_robots)

    def updateRobot(self, robot):
        robot_name = robot.robotName()
        
        # 先從當前集合中獲取機器人當前的分類
        current_classification = None
        if robot_name in self.horizontal_robots:
            current_classification = "horizontal"
        elif robot_name in self.vertical_robots:
            current_classification = "vertical"
        
        # 重新評估機器人的方向
        # 考慮精度問題，使用更寬容的比較
        x_diff = abs(robot.pos_x - self.coordinate.x)
        y_diff = abs(robot.pos_y - self.coordinate.y)
        
        is_main_intersection = self.coordinate.x == 15 and self.coordinate.y == 15
        new_classification = None
        
        # 確定新的分類
        if x_diff < 0.5:  # 垂直方向
            new_classification = "vertical"
        elif y_diff < 0.5:  # 水平方向
            new_classification = "horizontal"
        elif x_diff <= 3 and y_diff <= 3:  # 根據朝向決定
            if robot.heading == 0 or robot.heading == 180:  # 垂直方向
                new_classification = "vertical"
            elif robot.heading == 90 or robot.heading == 270:  # 水平方向
                new_classification = "horizontal"
        
        # 如果分類發生變化，需要將機器人從舊集合移動到新集合
        if current_classification != new_classification and new_classification is not None:
            # 首先從當前集合中移除
            if current_classification == "horizontal":
                del self.horizontal_robots[robot_name]
                if is_main_intersection:
                    print(f"Reclassification: Robot {robot_name} changed from horizontal to vertical (heading: {robot.heading})")
            elif current_classification == "vertical":
                del self.vertical_robots[robot_name]
                if is_main_intersection:
                    print(f"Reclassification: Robot {robot_name} changed from vertical to horizontal (heading: {robot.heading})")
            # 添加到新的集合
            if new_classification == "horizontal":
                self.horizontal_robots[robot_name] = robot
            elif new_classification == "vertical":
                self.vertical_robots[robot_name] = robot
        elif new_classification is not None:
            # 更新現有的機器人對象
            if new_classification == "horizontal":
                self.horizontal_robots[robot_name] = robot
            elif new_classification == "vertical":
                self.vertical_robots[robot_name] = robot

    def calculateAverageWaitingTimePerDirection(self, tick):
        total_waiting_time_horizontal = 0
        total_waiting_time_vertical = 0

        for robot in self.horizontal_robots.values():
            if robot.current_intersection_start_time is not None:
                total_waiting_time_horizontal += tick - robot.current_intersection_start_time

        for robot in self.vertical_robots.values():
            if robot.current_intersection_start_time is not None:
                total_waiting_time_vertical += tick - robot.current_intersection_start_time

        average_waiting_time_horizontal = total_waiting_time_horizontal / len(
            self.horizontal_robots) if self.horizontal_robots else 0
        average_waiting_time_vertical = total_waiting_time_vertical / len(
            self.vertical_robots) if self.vertical_robots else 0

        return average_waiting_time_horizontal, average_waiting_time_vertical

    def changeTrafficLight(self, direction, tick):
        if self.allowed_direction == direction:
            return
        self.allowed_direction = direction
        self.last_changed_tick = tick
        print(f"Intersection: {self.id} Changed allowed direction to {direction} at tick {tick} for intersection {self.id}")

    def isAllowedToMove(self, robot_heading):
        if self.allowed_direction is None:
            return True
        if robot_heading in (0, 180):
            return self.allowed_direction == 'Vertical'
        elif robot_heading in (90, 270):
            return self.allowed_direction == 'Horizontal'

    def getAllowedDirectionCode(self):
        if self.allowed_direction is None:
            return 0
        elif self.allowed_direction == 'Vertical':
            return 1
        elif self.allowed_direction == 'Horizontal':
            return 2

    @staticmethod
    def getAllowedDirectionByCode(code):
        if code == 0:
            return None
        elif code == 1:
            return 'Vertical'
        elif code == 2:
            return 'Horizontal'

    def printInfo(self):
        print("Current Allowed Direction:", self.allowed_direction)
        print("Last Updated Tick:", self.last_changed_tick)

    def addConnectedIntersectionId(self, x, y):
        intersection_id = f"{x}-{y}"
        self.connected_intersection_ids.append(intersection_id)

    def shouldSaveRobotInfo(self):
        if self.coordinate.x == 15:
            return True
        else:
            return False

    def setRLModelName(self, model_name):
        self.RL_model_name = f"IntersectionModel_{model_name}"

    def recordRobotPass(self, robot, current_tick, waiting_time=0):
        """
        記錄機器人通過交叉口的信息，用於計算流量和等待時間
        
        Args:
            robot: 通過的機器人
            current_tick: 當前的模擬時間
            waiting_time: 機器人在交叉口等待的時間
        """
        # 如果是第一個記錄，設置模擬開始時間
        if self.simulation_start_tick == 0:
            self.simulation_start_tick = current_tick
        
        # 記錄機器人通過
        self.total_robots_passed += 1
        self.robot_pass_records.append((robot.robotName(), current_tick))
        
        # 記錄等待時間
        self.total_waiting_time += waiting_time
        if waiting_time > 0:
            self.waiting_time_records.append((robot.robotName(), waiting_time))
        
        # 如果有調試輸出
        if robot.DEBUG_LEVEL >= 2:
            print(f"Robot {robot.robotName()} passed intersection {self.id} at tick {current_tick} with waiting time {waiting_time}")
            
    def getAverageTrafficRate(self, current_tick):
        """
        計算平均交叉口流量（單位時間內通過的機器人數量）
        
        Args:
            current_tick: 當前的模擬時間
            
        Returns:
            float: 平均交叉口流量（機器人/tick）
        """
        if self.simulation_start_tick == 0 or current_tick <= self.simulation_start_tick:
            return 0
            
        # 計算從模擬開始到現在的時間
        elapsed_time = current_tick - self.simulation_start_tick
        
        # 計算平均流量（機器人/tick）
        return self.total_robots_passed / elapsed_time if elapsed_time > 0 else 0
        
    def getAverageWaitingTime(self):
        """
        計算平均等待時間
        
        Returns:
            float: 平均等待時間（tick）
        """
        if not self.waiting_time_records:
            return 0
            
        total_waiting_time = sum(wait_time for _, wait_time in self.waiting_time_records)
        return total_waiting_time / len(self.waiting_time_records)
