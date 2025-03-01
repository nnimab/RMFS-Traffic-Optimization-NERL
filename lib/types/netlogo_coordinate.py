class NetLogoCoordinate:
    x = 0
    y = 0
    dimension = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "({},{})".format(self.x, self.y)
