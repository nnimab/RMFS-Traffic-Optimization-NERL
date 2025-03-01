from lib.types.netlogo_coordinate import NetLogoCoordinate
from lib.constant import *

class Object:
    def __init__(self, id: int, object_type: str, x: int, y: int):
        self.id = f"{object_type}-{id}"
        self.shape = 'empty-space'
        self.heading = 0
        self.velocity = 0
        self.acceleration = 0
        self.object_type = object_type
        self.pos_x = x
        self.pos_y = y
        self.coordinate = NetLogoCoordinate(x, y)
        self.color = 15  # red

    def move(self):
        if self.velocity != 0 or self.acceleration != 0:
            if self.acceleration != 0:
                self.velocity = self.velocity + (self.acceleration * TICK_TO_SECOND)

            if self.heading == 0:
                self.pos_y += self.velocity * TICK_TO_SECOND
            elif self.heading == 180:
                self.pos_y -= self.velocity * TICK_TO_SECOND
            elif self.heading == 90:
                self.pos_x += self.velocity * TICK_TO_SECOND
            elif self.heading == 270:
                self.pos_x -= self.velocity * TICK_TO_SECOND

    def rotate(self):
        self.heading += 90
        if self.heading == 360:
            self.heading == 0

    def rotateCC(self):
        if self.heading == 0:
            self.heading = 270
            return

        self.heading -= 90
