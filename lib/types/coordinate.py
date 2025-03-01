from lib.types.netlogo_coordinate import NetLogoCoordinate
class Coordinate:
    x = 0
    y = 0
    landscape = None

    def __init__(self, x=0, y=0, landscape=None):
        self.x = x
        self.y = y
        self.landscape = landscape

    def __repr__(self):
        return "({},{})".format(self.x, self.y)