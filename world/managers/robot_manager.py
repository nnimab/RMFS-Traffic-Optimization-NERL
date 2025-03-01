from __future__ import annotations
from typing import List, TYPE_CHECKING
from world.entities.robot import Robot
if TYPE_CHECKING:
    from world.warehouse import Warehouse

class RobotManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.robots: List[Robot] = []
        self.robot_counter = 0
    
    def initRobotManager(self):
        for robot in self.robots:
            robot.setRobotManager(self)
    
    def getAllRobots(self):
        return self.robots
    
    def getRobotByName(self, robot_name):
        for o in self.getAllRobots():
            if o.object_type == "robot" and o.robotName() == robot_name:
                return o

    def getRobotByCoordinate(self, x, y):
        for o in self.getAllRobots():
            if o.object_type == "robot" and o.pos_x == x and o.pos_y == y:
                return o
                    
    def getRobotsByCoordinate(self, coords):
        robots = []
        for coord in coords:
            robot = self.getRobotByCoordinate(coord[0], coord[1])
            if robot:
                robots.append(robot)
        return robots
    
    def createRobot(self, x, y):
        robot = Robot(self.robot_counter, x, y)
        self.robots.append(robot)
        self.robot_counter += 1
        robot._id = self.warehouse.total_pod + 1
        self.warehouse.total_pod += 1
        return robot
