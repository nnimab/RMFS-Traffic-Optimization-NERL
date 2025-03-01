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

    def setIntersectionManager(self, intersection_manager):
        self.intersection_manager = intersection_manager

    def durationSinceLastChange(self, tick):
        return tick - self.last_changed_tick

    def addRobot(self, robot):
        if robot.pos_x == self.coordinate.x:
            self.horizontal_robots[robot.robotName()] = robot
        elif robot.pos_y == self.coordinate.y:
            self.vertical_robots[robot.robotName()] = robot

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
        if robot_name in self.horizontal_robots:
            self.horizontal_robots[robot_name] = robot
        elif robot_name in self.vertical_robots:
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
