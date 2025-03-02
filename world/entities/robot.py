import math
from typing import Optional, List, TYPE_CHECKING

from lib.types.heading import Heading
from lib.types.netlogo_coordinate import NetLogoCoordinate
from world.entities.object import Object
from lib.file import *
from world.entities.intersection import Intersection
from world.entities.job import Job
from .station import Station
from world.entities.zone import Zone
from lib.constant import *
if TYPE_CHECKING:
    from world.managers.robot_manager import RobotManager

class Robot(Object):
    # movement related
    MAXIMUM_SPEED = 1.5
    # energy consumption related
    MASS = 1
    LOAD_MASS = 0
    GRAVITY = 10
    FRICTION = 0.3
    INERTIA = 0.4

    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, 'robot', x, y)
        self._id = 0 # netlogo related
        self.robot_manager: RobotManager = None
        self.shape = 'turtle-2'
        self.turning = 0
        self.current_state = 'idle'
        self.energy_consumption = 0
        self.traffic_policy = []
        self.latest_tick = 0
        self.route_stop_points = []
        self.job: Optional[Job] = None
        self.turning_delay = 0
        self.taking_pod_delay = 0
        self.delay_per_task = 10
        self.idle_time = 0
        self.current_intersection_id = None
        self.future_intersection_id = None
        self.previous_intersection_id = None
        self.current_intersection_energy_consumption = 0
        self.current_intersection_stop_and_go = 0
        self.current_intersection_start_time = None
        self.current_intersection_finish_time = None

    def setRobotManager(self, robot_manager):
        self.robot_manager = robot_manager

    @staticmethod
    def _checkMovementDirection(p1, p2):
        if p1.y == p2.y:
            # horizontal
            if p1.x < p2.x:
                return 90
            if p1.x > p2.x:
                return 270

        if p1.x == p2.x:
            # vertical
            if p1.y < p2.y:
                return 0
            if p1.y > p2.y:
                return 180

        return None
    
    def calculateEnergy(self, velocity, acceleration):
        tick_unit = TICK_TO_SECOND
        if acceleration != 0 and velocity != 0:
            average_speed = 2 * velocity + (acceleration * tick_unit)
            return (self.MASS + self.LOAD_MASS) * ((self.GRAVITY * self.FRICTION) + (
                    acceleration * self.INERTIA)) * average_speed * tick_unit / 7200
        elif velocity != 0:
            return (self.MASS + self.LOAD_MASS) * self.GRAVITY * self.FRICTION * velocity * tick_unit / 3600
        return 0

    def setPath(self, path):
        current_heading = self.heading
        route_stop_points = []

        # convert path list to route_stop_points (list of NetLogoCoordinate where the robot should stop and Heading to turn the robot)
        for i in range(1, len(path), 1):
            p1 = NetLogoCoordinate(path[i - 1][0], path[i - 1][1])
            p2 = NetLogoCoordinate(path[i][0], path[i][1])
            heading = self.getHeading(p1, p2)
            if current_heading != heading:
                current_heading = heading
                route_stop_points.append(NetLogoCoordinate(path[i - 1][0], path[i - 1][1]))
                route_stop_points.append(Heading(heading))
        if len(route_stop_points) > 0 and isinstance(route_stop_points[len(route_stop_points) - 1],
                                                     NetLogoCoordinate) == False:
            now = path[len(path) - 1]
            route_stop_points.append(NetLogoCoordinate(now[0], now[1]))
        self.route_stop_points = route_stop_points

    def changeColorByState(self):
        if self.current_state == "taking_pod":
            self.color = 57  # green
        elif self.current_state == "delivering_pod":
            self.color = 15  # red
        elif self.current_state == "returning_pod":
            self.color = 46  # yellow
        elif self.current_state == "station_processing":
            self.color = 94  # brown
        elif self.current_state == "idle":
            self.color = 0  # black

    def advanceState(self):
        if self.current_state == "taking_pod":
            self.taking_pod_delay += self.delay_per_task
            self.current_state = "delivering_pod"
        elif self.current_state == "delivering_pod":
            self.current_state = "station_processing"
        elif self.current_state == "station_processing":
            self.current_state = "returning_pod"
        elif self.current_state == "returning_pod":
            self.taking_pod_delay += self.delay_per_task
            self.current_state = "idle"

    def decideCollision(self, collision_block, o, collide_distance):
        will_collide = False
        selected_label = ""
        object_heading = 0
        distance = math.sqrt((o['x'] - self.pos_x) ** 2 + (o['y'] - self.pos_y) ** 2)
        if collision_block is not None:
            self_distance = self._calculateTwoPoint(NetLogoCoordinate(self.pos_x, self.pos_y),
                                                    NetLogoCoordinate(collision_block[0], collision_block[1]))
            if (self.velocity ** 2) / 2 >= self_distance or distance < collide_distance:
                will_collide = True
                selected_label = o['label']
                object_heading = o['heading']
        elif distance < collide_distance:
            will_collide = True
            selected_label = o['label']
            object_heading = o['heading']
        return will_collide, selected_label, object_heading

    def getNearestRobotConflictCandidate(self, next_step_coords, search_area):
        neighbor_candidates = []

        # 安全檢查：確保坐標在有效範圍內
        pos_x = round(self.pos_x)
        pos_y = round(self.pos_y)
        warehouse_dimension = self.robot_manager.warehouse.DIMENSION
        if pos_x < 0 or pos_x >= warehouse_dimension or pos_y < 0 or pos_y >= warehouse_dimension:
            # 如果坐標超出範圍，返回None避免潛在錯誤
            return None

        neighbors = self.robot_manager.warehouse.landscape.getNeighborObjectWithRadius(pos_x, pos_y, search_area)
        if neighbors:
            for neighbor in neighbors:
                neighbor_robot = self.robot_manager.getRobotByName(neighbor['label'])
                if neighbor_robot == self:
                    continue

                # if neighbor['velocity'] == 0:
                #     continue

                neighbor_next_step_coords = self._calculateNextBlocks(round(neighbor['x']), round(neighbor['y']),
                                                                    neighbor['heading'], search_area)
                meeting_coordinate = self._getIntersectionBlock(next_step_coords, neighbor_next_step_coords)
                if meeting_coordinate and self.isCollisionCandidate(neighbor):
                    neighbor_candidates.append((neighbor, meeting_coordinate))

        return neighbor_candidates

    def getPriorityDiff(self, object):
        state_priority = {
            'station_processing': 3,
            'delivering_pod': 3,
            'returning_pod': 2,
            'taking_pod': 1,
            'idle': 0
        }

        self_priority = state_priority[self.current_state]
        other_priority = state_priority[object['state']]

        return self_priority - other_priority

    def isCollisionCandidate(self, obj):
        # Check for collision candidate based on relative positions and headings
        relative_x = obj['x'] - self.pos_x
        relative_y = obj['y'] - self.pos_y

        if self.heading == 0:
            return relative_y >= 0
        if self.heading == 180:
            return relative_y <= 0
        if self.heading == 90:
            return relative_x >= 0
        if self.heading == 270:
            return relative_x <= 0

    def shouldRemovePolicy(self, prioritized_robot, self_blocks, other_blocks):
        if self.heading == prioritized_robot.heading:
            return self.velocity < prioritized_robot.velocity
        else:
            return self._getIntersectionBlock(self_blocks, other_blocks) is None

    @staticmethod
    def _transformRouteToList(path):
        path_int = []
        for p in path:
            l = p.split(',')
            path_int.append([int(l[0]), int(l[1])])
        return path_int

    @staticmethod
    def transformCoordinatesToList(coords: List[NetLogoCoordinate]):
        path_int = []
        for coord in coords:
            path_int.append([coord.x, coord.y])

        return path_int

    def neutralizeRobotState(self):
        self.route_stop_points = []

    def updateCurrentPosition(self):
        self.coordinate = NetLogoCoordinate(round(self.pos_x), round(self.pos_y))
        self.pos_x = round(self.pos_x)
        self.pos_y = round(self.pos_y)

    def pickingItemInPod(self):
        if self.job is not None and self.isBeingProcessOnStation():
            self.job.decrementDelay()
            return True

    def isInStationPath(self):
        if self.job is not None:
            station: Station = self.robot_manager.warehouse.station_manager.getStationById(self.job.station_id)
            for coord in station.getPath():
                if round(self.pos_x) == coord.x and round(self.pos_y) == coord.y:
                    return True

    def isBeingProcessOnStation(self):
        station: Station = self.robot_manager.warehouse.station_manager.getStationById(self.job.station_id)
        return self.job.isBeingProcessed() and self.closeEnough(
            station.coordinate, 0.1)

    def movementPlan(self):
        if self.pickingItemInPod():
            return

        if not self.route_stop_points:
            self.advanceStateIfNeeded()
            return

        if self.eligibleToReroute():
            if self.current_state == "taking_pod":
                self.setMove(self.route_stop_points[-1], self.robot_manager.warehouse.graph, avoid_side=True)
            elif self.current_state == "delivering_pod" or self.current_state == "returning_pod":
                self.setMove(self.route_stop_points[-1], self.robot_manager.warehouse.graph_pod, avoid_side=True)
            elif self.current_state == "station_processing":
                station: Station = self.robot_manager.warehouse.station_manager.getStationById(self.job.station_id)
                station.updateRobotRouteType(self.robotName())
                path = station.getSubPath(self.robotName(), round(self.pos_x), round(self.pos_y))
                self.setPath(self.transformCoordinatesToList(path))

            self.idle_time = 0

        if not self.route_stop_points:
            return

        next_destination_coordinate = self.route_stop_points[0]

        if isinstance(next_destination_coordinate, Heading):
            self.handleDirectional(next_destination_coordinate)
            return

        if self.notAbleToMove(next_destination_coordinate):
            self.updateIdleState()
            return

        candidate_conflict_coordinate = self.handleConflicts(next_destination_coordinate)

        self.executeMove(candidate_conflict_coordinate, next_destination_coordinate)

    def updateIdleState(self):
        self.idle_time += 1
        self.velocity = 0
        self.acceleration = 0
        self.robot_manager.warehouse.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity,
                                          self.acceleration, self.heading, self.current_state)

        if self.current_intersection_id:
            self.current_intersection_stop_and_go += 1

    def handleConflicts(self, next_destination_coordinate):
        candidate_conflict_coordinate = None
        if isinstance(next_destination_coordinate, NetLogoCoordinate) and not self.isInStationPath():
            self_coord = NetLogoCoordinate(self.pos_x, self.pos_y)
            search_area = self.calculateSearchArea(next_destination_coordinate)

            next_step_coordinates = self._calculateNextBlocks(round(self.pos_x), round(self.pos_y),
                                                                self.heading, search_area)
            nearest_conflict_candidates = self.getNearestRobotConflictCandidate(next_step_coordinates, search_area)
            if nearest_conflict_candidates is not None:
                for candidate, meeting_coordinate in nearest_conflict_candidates:
                    if (candidate['state'] == "station_processing" or candidate['state'] == 'idle'
                            or candidate['velocity'] == 0):
                        continue

                    neighbor_coord = NetLogoCoordinate(candidate['x'], candidate['y'])
                    meeting_coordinate = NetLogoCoordinate(meeting_coordinate[0], meeting_coordinate[1])
                    self_distance_to_meeting_block = self._calculateTwoPoint(self_coord, meeting_coordinate)
                    neighbor_distance_to_meeting_block = self._calculateTwoPoint(neighbor_coord, meeting_coordinate)

                    priority_diff = self.getPriorityDiff(candidate)
                    if candidate['heading'] == self.heading:
                        for x, y in next_step_coordinates:
                            if round(candidate['x']) == x and round(candidate['y']) == y:
                                candidate_conflict_coordinate = self.calculateNextMovementFromConflict(
                                    meeting_coordinate, next_destination_coordinate)
                                break

                    elif priority_diff < 0:
                        candidate_conflict_coordinate = self.calculateNextMovementFromConflict(meeting_coordinate,
                                                                                                   next_destination_coordinate)

                    elif priority_diff == 0:
                        if self_distance_to_meeting_block > neighbor_distance_to_meeting_block:
                            candidate_conflict_coordinate = self.calculateNextMovementFromConflict(
                                meeting_coordinate, next_destination_coordinate)

        return candidate_conflict_coordinate

    def executeMove(self, candidate_conflict_coordinate, next_destination_coordinate):
        self.idle_time = 0
        if candidate_conflict_coordinate and candidate_conflict_coordinate != next_destination_coordinate:
            self.handleNextMovement(candidate_conflict_coordinate, is_next_route_stop=False)
        else:
            self.handleNextMovement(next_destination_coordinate, is_next_route_stop=True)

        self.drawNextPosition()

    def eligibleToReroute(self):
        if self.idle_time <= 50 or self.current_state == "delivering_pod":
            return False

        if self.isInStationPath():
            station: Station = self.robot_manager.warehouse.station_manager.getStationById(self.job.station_id)

            if self.current_state == "station_processing" and station.hasRouteChanged(self.robotName()):
                return True
            else:
                return False

        # Calculate next step coordinates
        next_step_coordinates = self._calculateNextBlocks(
            round(self.pos_x), round(self.pos_y), self.heading, 1, include_self=False)
        robot_front = self.robot_manager.warehouse.landscape.getNeighborObject(*next_step_coordinates[0])

        # Check if there is no robot in front
        if not robot_front:
            return False

        # Check if the robot in front is idle
        if robot_front['state'] == "idle":
            return True

        # Check if the robot in front is heading in the same direction
        if robot_front['heading'] == self.heading:
            return False

        # Compare priorities
        priority_diff = self.getPriorityDiff(robot_front)
        if priority_diff > 0:
            return False
        elif priority_diff < 0:
            return True

        # Resolve ties by ID
        return self.robotID(self.robotName()) < self.robotID(robot_front['label'])

    def calculateNextMovementFromConflict(self, conflict_coordinate: NetLogoCoordinate,
                                              next_destination_coordinate: NetLogoCoordinate):
        potential_next = None
        if self.heading == 0:
            potential_next = NetLogoCoordinate(conflict_coordinate.x, conflict_coordinate.y - 1)
        elif self.heading == 180:
            potential_next = NetLogoCoordinate(conflict_coordinate.x, conflict_coordinate.y + 1)
        elif self.heading == 90:
            potential_next = NetLogoCoordinate(conflict_coordinate.x - 1, conflict_coordinate.y)
        elif self.heading == 270:
            potential_next = NetLogoCoordinate(conflict_coordinate.x + 1, conflict_coordinate.y)

        if self.heading in [0, 180]:
            if abs(potential_next.y - next_destination_coordinate.y) < abs(
                    conflict_coordinate.y - next_destination_coordinate.y):
                return next_destination_coordinate
        else:
            if abs(potential_next.x - next_destination_coordinate.x) < abs(
                    conflict_coordinate.x - next_destination_coordinate.x):
                return next_destination_coordinate

        return potential_next

    def calculateSearchArea(self, next_destination_coordinate: NetLogoCoordinate):
        if self.isInStationPath():
            return 1

        return 3

    def notAbleToMove(self, next_destination_coordinate: NetLogoCoordinate):
        if self.turning_delay > 0:
            self.turning_delay -= 1
            return True

        if self.taking_pod_delay > 0:
            self.taking_pod_delay -= 1
            return True

        if next_destination_coordinate.x == round(self.pos_x) and next_destination_coordinate.y == round(self.pos_y):
            return False

        return self.pathBlocked()

    def pathBlocked(self):
        next_step_coordinates = self._calculateNextBlocks(round(self.pos_x), round(self.pos_y),
                                                            self.heading, 1, include_self=False)

        if not self.isAlignedWithHeading(next_step_coordinates):
            return False

        return (self.pathBlockedByIntersection(next_step_coordinates)
                or self.pathBlockedByRobot(next_step_coordinates))

    def pathBlockedByIntersection(self, next_step_coordinates):
        for next_x, next_y in next_step_coordinates:
            intersection = self.robot_manager.warehouse.intersection_manager.getIntersectionByCoordinate(next_x, next_y)
            if (intersection and self.closeEnough(intersection.coordinate, 1)
                    and not intersection.isAllowedToMove(self.heading)):
                return True
        return False

    def pathBlockedByRobot(self, next_step_coordinates):
        neighbors = self.robot_manager.warehouse.landscape.getNeighborObjectWithRadius(round(self.pos_x), round(self.pos_y), 2)
        for neighbor in neighbors:
            if self.robot_manager.getRobotByName(neighbor['label']) == self:
                continue

            near_robot_coord = NetLogoCoordinate(neighbor['x'], neighbor['y'])

            for next_x, next_y in next_step_coordinates:
                x_difference = abs(neighbor['x'] - next_x)
                y_difference = abs(neighbor['y'] - next_y)
                if x_difference < 1 and y_difference < 1 and self.closeEnough(near_robot_coord, 1):
                    self_distance_to_conflict = abs(self.pos_x - next_x) + abs(self.pos_y - next_y)
                    neighbor_robot_distance_to_conflict = abs(neighbor['x'] - next_x) + abs(neighbor['y'] - next_y)

                    if neighbor_robot_distance_to_conflict < self_distance_to_conflict:
                        return True
                    else:
                        continue

        return False

    def isInPath(self, destination, steps):
        # Assuming steps is a list of tuples (x, y) coordinates
        current_x, current_y = round(self.pos_x), round(self.pos_y)
        dest_x, dest_y = destination.x, destination.y

        # Check if destination is on the path between current and next step
        for step_x, step_y in steps:
            if self.heading == 0:  # Moving up along the y-axis
                if current_y <= dest_y and current_x == dest_x:
                    return current_x == step_x and current_y <= step_y <= dest_y
            elif self.heading == 180:  # Moving down along the y-axis
                if current_y >= dest_y and current_x == dest_x:
                    return current_x == step_x and current_y >= step_y >= dest_y
            elif self.heading == 90:  # Moving right along the x-axis
                if current_x <= dest_x and current_y == dest_y:
                    return current_y == step_y and current_x <= step_x <= dest_x
            elif self.heading == 270:  # Moving left along the x-axis
                if current_x >= dest_x and current_y == dest_y:
                    return current_y == step_y and current_x >= step_x >= dest_x
        return False

    def isAlignedWithHeading(self, steps):
        for step_x, step_y in steps:
            if self.heading in (0, 180):  # Vertical movement
                return round(self.pos_x) == step_x
            elif self.heading in (90, 270):  # Horizontal movement
                return round(self.pos_y) == step_y

    def handleDirectional(self, heading):
        angular_change = min(abs(heading.getHeading() - self.heading),
                             360 - abs(heading.getHeading() - self.heading)) // 90
        self.turning_delay += self.delay_per_task * angular_change

        self.heading = heading.getHeading()
        self.turning += 1
        self.route_stop_points.pop(0)

    def handleNextMovement(self, next_destination_coordinate, is_next_route_stop=True):
        current_coord = NetLogoCoordinate(self.pos_x, self.pos_y)

        if self.closeEnough(next_destination_coordinate, 0.3):
            self.updatePosition(next_destination_coordinate)
            if is_next_route_stop:
                self.route_stop_points.pop(0)
        else:
            self.updateMotionParameters(current_coord, next_destination_coordinate)

    def updatePosition(self, coordinate):
        # Update robot's position and movement parameters to match the intersection_coordinate
        self.pos_x = round(coordinate.x)
        self.pos_y = round(coordinate.y)
        self.coordinate = NetLogoCoordinate(self.pos_x, self.pos_y)
        self.velocity = 0
        self.acceleration = 0

    def advanceStateIfNeeded(self):
        # Check if all route points are done and handle state transitions
        if not self.route_stop_points:
            self.advanceState()
            self.updateCurrentPosition()
            if self.current_state == "delivering_pod":
                self.setMoveToStationGate()
            elif self.current_state == "returning_pod":
                station: Station = self.robot_manager.warehouse.station_manager.getStationById(self.job.station_id)
                station.removeRobot(self.robotName())
                self.setMove(self.job.pod_coordinate, self.robot_manager.warehouse.graph_pod, need_neutralize_robot=True)
            elif self.current_state == "station_processing":
                station: Station = self.robot_manager.warehouse.station_manager.getStationById(self.job.station_id)
                station.addRobot(self.robotName())
                self.setPath(self.transformCoordinatesToList(station.getRobotRoute(self.robotName())))

        self.robot_manager.warehouse.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity, self.acceleration,
                                          self.heading, self.current_state)

    def updateMotionParameters(self, current_coord, next_destination_coordinate):
        # Adjust robot's acceleration based on proximity to the next intersection_coordinate
        self.acceleration = 1
        deceleration_buffer = 0.5
        distance_to_stop = self._calculateTwoPoint(current_coord, next_destination_coordinate)
        if (self.velocity ** 2) / (2 * deceleration_buffer) >= distance_to_stop:
            self.acceleration = -1

    def move(self):
        self.changeColorByState()
        self.movementPlan()
        self.latest_tick += 1

    def drawNextPosition(self):
        initial_velocity = self.velocity
        initial_acceleration = self.acceleration

        energy = self.calculateEnergy(initial_velocity, initial_acceleration)
        self.energy_consumption += energy

        if self.velocity != 0:
            distance_delta = self.velocity * TICK_TO_SECOND
            if self.heading == 0:
                self.pos_y += distance_delta
            elif self.heading == 180:
                self.pos_y -= distance_delta
            elif self.heading == 90:
                self.pos_x += distance_delta
            elif self.heading == 270:
                self.pos_x -= distance_delta
        self.coordinate = NetLogoCoordinate(round(self.pos_x), round(self.pos_y))

        if self.acceleration != 0:
            self.velocity += (self.acceleration * TICK_TO_SECOND)
            self.velocity = max(0, min(self.MAXIMUM_SPEED, self.velocity))

        # for traffic policy purposes, report states to the manager
        self.robot_manager.warehouse.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity, self.acceleration,
                                          self.heading, self.current_state)
        self.updateIntersectionInformation(energy)

    def updateIntersectionInformation(self, energy):
        intersection_id = self.robot_manager.warehouse.intersection_manager.findIntersectionByCoordinate(round(self.pos_x),
                                                                                                  round(self.pos_y))
        if intersection_id:
            if self.current_intersection_id == intersection_id:
                self.updateCurrentIntersection(energy)
            else:
                self.finalizeCurrentIntersection()

                self.startNewIntersection(intersection_id)
        else:
            self.finalizeCurrentIntersection()

            self.resetIntersectionTracking()

    def updateCurrentIntersection(self, energy):
        self.current_intersection_energy_consumption += energy

        intersection: Intersection = self.robot_manager.warehouse.intersection_manager.findIntersectionById(
            self.current_intersection_id)
        intersection.updateRobot(self)

    def finalizeCurrentIntersection(self):
        if self.current_intersection_id is None:
            return

        # Mark the finish time for the current intersection
        self.current_intersection_finish_time = self.robot_manager.warehouse._tick
        # Log the intersection information to CSV

        intersection: Intersection = self.robot_manager.warehouse.intersection_manager.findIntersectionById(
            self.current_intersection_id)

        if intersection.shouldSaveRobotInfo():
            self.intersectionToCsv(intersection)

        intersection.removeRobot(self)

    def startNewIntersection(self, intersection_id):
        # Set the new intersection ID and reset the energy consumption
        self.current_intersection_id = intersection_id
        self.current_intersection_energy_consumption = 0
        self.current_intersection_start_time = self.robot_manager.warehouse._tick

        intersection: Intersection = self.robot_manager.warehouse.intersection_manager.findIntersectionById(
            self.current_intersection_id)
        intersection.addRobot(self)

    def resetIntersectionTracking(self):
        # Reset all intersection-related data
        self.current_intersection_id = None
        self.current_intersection_energy_consumption = 0
        self.current_intersection_stop_and_go = 0
        self.current_intersection_start_time = 0
        self.current_intersection_finish_time = 0

    def intersectionToCsv(self, intersection: Intersection):
        header = ["robot_name", "robot_state", "robot_destination", "intersection_start_time",
                  "intersection_finish_time", "intersection_id",
                  "energy_consumption_intersection", "queueing_robot"]
        data = [self.robotName(), self.current_state, self.route_stop_points[-1], self.current_intersection_start_time,
                self.current_intersection_finish_time, self.current_intersection_id,
                self.current_intersection_energy_consumption,
                intersection.robotCount()]

        write_to_csv("intersection-energy-consumption.csv", header, data,
                     self.robot_manager.warehouse.landscape.current_date_string)

    def assignJobAndSetToTakePod(self, job: Job):
        self.job = job

        self.setMoveToTakePod()

    def assignJobAndSetToStation(self, job: Job):
        self.job = job
        self.current_state = "taking_pod"
        self.route_stop_points = None
        self.advanceStateIfNeeded()
        self.taking_pod_delay = 0

    def setMoveToTakePod(self):
        self.setMove(self.job.pod_coordinate, graph=self.robot_manager.warehouse.graph, need_neutralize_robot=False)
        self.current_state = "taking_pod"

    def setMoveToStationGate(self):
        station: Station = self.robot_manager.warehouse.station_manager.getStationById(self.job.station_id)
        self.setMove(station.getPath()[0], graph=self.robot_manager.warehouse.graph_pod, need_neutralize_robot=False)

    def setMove(self, dest: NetLogoCoordinate, graph, need_neutralize_robot: bool = False, avoid_side: bool = False):
        start = self.coordinateToStringKey(round(self.pos_x), round(self.pos_y))
        end = self.coordinateToStringKey(dest.x, dest.y)

        if need_neutralize_robot:
            self.neutralizeRobotState()

        nodes_to_avoid = []
        if avoid_side:
            avoid_coords = self.calculateAllDirectionNextBlocks(round(self.pos_x), round(self.pos_y), 1,
                                                                     include_self=False)
            for avoid_coord in avoid_coords:
                if self.robot_manager.warehouse.landscape.getNeighborObject(*avoid_coord) is None:
                    continue

                nodes_to_avoid.append(self.coordinateToStringKey(*avoid_coord))

        node_routes = None
        if self.robot_manager.warehouse.zoning:
            zone_boundary, penalties = self.createZone(method="kmeans")
            node_routes = graph.dijkstraModified(start,end, penalties, zone_boundary, nodes_to_avoid)
        else:
            node_routes = graph.dijkstra(start, end, nodes_to_avoid) # This one is baseline
        
        self.setPath(self._transformRouteToList(node_routes))

    def createZone(self, method):
        robot_objects = self.robot_manager.warehouse.landscape.getRobotObject()
        robots_location = [[info['x'], info['y']] for info in robot_objects.values() if info['state'] != 'station_processing']
        robots_idle_time = []
        robot_list = []
        if len(robots_location) > 0:
            robot_list = self.robot_manager.getRobotsByCoordinate(robots_location)

        for robot in robot_list:
            robots_idle_time.append(robot.idle_time)
        
        zones = self.robot_manager.warehouse.zone_manager.createZone(robots_location, self.robot_manager.warehouse.getWarehouseSize(), methods=method)
        penalties = zones.calculatePenalty(robots_location, robots_idle_time, self.robot_manager.warehouse.getWarehouseSize(), threshold=5)
        zone_boundary = zones.getBoundary()
        return zone_boundary, penalties

    # utility functions
    
    @staticmethod
    def calculateAllDirectionNextBlocks(x, y, block_count=5, include_self=False):
        headings = [0, 90, 180, 270]
        result = []

        for heading in headings:
            blocks = Robot._calculateNextBlocks(x, y, heading, block_count, include_self)
            for block in blocks:
                result.append(block)

        return result
    
    @staticmethod
    def getHeading(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        if p1.x == p2.x:
            if p1.y > p2.y:
                return 180
            else:
                return 0
        elif p1.y == p2.y:
            if p1.x > p2.x:
                return 270
            else:
                return 90

    @staticmethod
    def _calculateTwoPoint(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def closeEnough(self, p: NetLogoCoordinate, precision=0.25):
        self_coord = NetLogoCoordinate(self.pos_x, self.pos_y)
        return self._calculateTwoPoint(self_coord, p) < precision

    @staticmethod
    def ensureCoordinate(number):
        if isinstance(number, int):
            print(f"{number} is an integer.")
        elif isinstance(number, float) and number.is_integer():
            print(f"{number} is a float with 0 precision.")
        else:
            print(f"{number} is not a valid integer or float with 0 precision.")

    @staticmethod
    def getDecimal(number):
        subtractor = int(number)
        return number - subtractor

    def robotName(self):
        return f"robot-{self._id}"

    @staticmethod
    def robotID(robot_name):
        return int(robot_name.split('-')[1])

    @staticmethod
    def calculateAllDirectionNextBlocks(x, y, block_count=5, include_self=False):
        headings = [0, 90, 180, 270]
        result = []

        for heading in headings:
            blocks = Robot._calculateNextBlocks(x, y, heading, block_count, include_self)
            for block in blocks:
                result.append(block)

        return result

    @staticmethod
    def _calculateNextBlocks(x, y, heading, block_count=5, include_self=False):
        x_difference = 0
        y_difference = 0

        if heading == 0:
            y_difference = 1
        if heading == 90:
            x_difference = 1
        if heading == 270:
            x_difference = -1
        if heading == 180:
            y_difference = -1

        result = []

        if include_self:
            result.append([x, y])
        for i in range(block_count):
            x += x_difference
            y += y_difference

            result.append([x, y])

        return result

    @staticmethod
    def coordinateToStringKey(x: int, y: int):
        return "{},{}".format(x, y)

    @staticmethod
    def _getIntersectionBlock(blocks_1, blocks_2):
        for p in blocks_1:
            if p in blocks_2:
                return p
