import csv
import random

from lib.file import *   
from lib.constant import *   

class Layout(object):
    def __init__(self):
        self.pod_batch_horizontal = 5
        self.pod_batch_vertical = 2
        self.pod_batch_horizontal_max = 5
        self.pod_batch_vertical_count = 0
        self.pod_batch_vertical_max = 10
        self.reserved_column_start = 9
        self.reserved_column_end = 9
        self.reserved_column_station = 5
        self.order_picker_total = 3
        self.order_replenishment_total = 2
        self.horizontal_direction_switch = False
        self.vertical_direction_switch = False
        self.total_pods_active = 300 # Number of pods
        self.total_charging_stations = 10

    def generate(self):
        order_picker_positions = self.calculateStationPositions(self.order_picker_total)
        order_replenishment_positions = self.calculateStationPositions(self.order_replenishment_total)

        data_matrix = []

        for row in range(self.totalRows()):
            current_row = []
            self.vertical_direction_switch = False
            for col in range(self.totalCols()):
                value_to_append = self.appendStationValue(row, col, order_picker_positions,
                                                            order_replenishment_positions)
                if value_to_append is not None:
                    current_row.append(value_to_append)
                else:
                    if self.reserved_column_start <= col < (self.totalCols() - self.reserved_column_end):
                        if row % (self.pod_batch_vertical + 1) == 0:
                            if (col - self.reserved_column_start) % (self.pod_batch_horizontal + 1) == 0:
                                current_row.append(3)
                                self.vertical_direction_switch = not self.vertical_direction_switch
                            else:
                                current_row.append(4 if self.horizontal_direction_switch else 5)
                        else:
                            pod_index = (col - self.reserved_column_start - 1) % (self.pod_batch_horizontal + 1)
                            if pod_index < self.pod_batch_horizontal:
                                current_row.append(1)
                            else:
                                current_row.append(6 if self.vertical_direction_switch else 7)
                                self.vertical_direction_switch = not self.vertical_direction_switch
                    else:
                        current_row.append(6 if self.vertical_direction_switch else 7)
                        self.vertical_direction_switch = not self.vertical_direction_switch

            data_matrix.append(current_row)
            self.horizontal_direction_switch = not self.horizontal_direction_switch

        self.adjustPodAvailability(data_matrix)

        pod_path = PARENT_DIRECTORY + '/data/output/generated_pod.csv'
        with open(pod_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_matrix)

    def appendStationValue(self, row, col, order_positions, replenish_positions):
        if col < self.reserved_column_station:
            return self.getValueForStation(row, col, order_positions)
        elif col >= self.totalCols() - self.reserved_column_station:
            return self.getValueForStation(row, col, replenish_positions, start_col=self.totalCols() - 1,
                                              mirrored=True)
        else:
            return None

    def calculateStationPositions(self, total_stations):
        return self.getStationRowIndexes(self.getStationPositions(total_stations))

    def totalRows(self):
        return (self.pod_batch_vertical_max * self.pod_batch_vertical) + self.pod_batch_vertical_max + 1

    def totalCols(self):
        # Total columns include reserved columns, pods, and spaces between pods
        total_pod_space = (self.pod_batch_horizontal * self.pod_batch_horizontal_max) + self.pod_batch_horizontal_max - 1
        return (self.reserved_column_start + 1) + total_pod_space + (self.reserved_column_end + 1)

    def determineStationLimits(self):
        return int((self.pod_batch_vertical_max + 1) / 2)

    def getStationPositions(self, total_station):
        total_numbers = self.determineStationLimits()
        numbers = list(range(1, total_numbers + 1))
        selected_sequence = []
        middle_index = len(numbers) // 2

        for i in range(total_station):
            if i % 2 == 0:
                offset = (i // 2)
            else:
                offset = -(i // 2) - 1

            selected_sequence.append(numbers[middle_index + offset])

        return sorted(selected_sequence)

    def getStationRowIndexes(self, order_picker_positions):
        result = []
        for i in order_picker_positions:
            result.append(self.getStationRowIndex(i))

        return result

    def getStationRowIndex(self, order_picker_position):
        start_index = (order_picker_position - 1) * (2 * self.pod_batch_vertical + 2)
        return start_index, start_index + self.pod_batch_vertical + 1

    @staticmethod
    def getValueForStation(current_row, current_col, ranges, start_col=0, mirrored=False):
        blank_space = 99
        rail_triangle = 24 if mirrored else 14
        rail_0 = 22 if mirrored else 12
        rail_1 = 23 if mirrored else 13
        corner_0 = 26 if mirrored else 16
        corner_1 = 27 if mirrored else 17
        corner_2 = 28 if mirrored else 18
        corner_3 = 29 if mirrored else 19
        station_picker_value = 21 if mirrored else 11

        offset = -1 if mirrored else 1

        for start, end in ranges:
            if current_row == start:
                # First row
                if current_col == start_col + 2 * offset:
                    return corner_0
                elif current_col == start_col + 3 * offset or current_col == start_col + 4 * offset:
                    return rail_0
                else:
                    return blank_space
            elif current_row == end:
                # Last row
                if current_col == start_col + 2 * offset:
                    return corner_1
                elif current_col == start_col + 3 * offset or current_col == start_col + 4 * offset:
                    return rail_1
                else:
                    return blank_space
            elif start < current_row < end:
                # Middle row
                if current_row == start + 1:
                    if current_col == start_col + 1 * offset:
                        return station_picker_value
                    elif current_col == start_col + 2 * offset:
                        return rail_triangle
                    elif current_col == start_col + 3 * offset:
                        return rail_1
                    elif current_col == start_col + 4 * offset:
                        return corner_2
                    else:
                        return blank_space
                if current_row == start + 2:
                    if current_col == start_col + 2 * offset:
                        return rail_triangle
                    elif current_col == start_col + 3 * offset:
                        return rail_0
                    elif current_col == start_col + 4 * offset:
                        return corner_3
                    else:
                        return blank_space
                else:
                    return blank_space
        return blank_space

    def adjustPodAvailability(self, matrix):
        # Count the current total number of active pods
        current_total_pods = sum(row.count(1) for row in matrix)

        # Calculate how many pods need to be deactivated and converted
        if current_total_pods > self.total_pods_active:
            pods_to_deactivate = current_total_pods - (self.total_pods_active + self.total_charging_stations)
            pods_to_convert = self.total_charging_stations
        else:
            pods_to_deactivate = 0
            pods_to_convert = 0

        # List of all pod positions in matrix that are currently active
        pod_positions = [(r, c) for r in range(len(matrix)) for c in range(len(matrix[r])) if matrix[r][c] == 1]

        # Randomly select and deactivate pods if there are any to deactivate
        if pods_to_deactivate > 0:
            for r, c in random.sample(pod_positions, pods_to_deactivate):
                matrix[r][c] = 0  # Mark this position as deactivated
                pod_positions.remove((r, c))  # Remove this position from available pods

        # Randomly select and convert remaining active pods to charging stations if needed
        if pods_to_convert > 0:
            for r, c in random.sample(pod_positions, pods_to_convert):
                matrix[r][c] = 2  # Mark this position as a charging station
