class Movement:
    acceleration = 0
    tick = 0

    def __repr__(self):
        return "acceleration: {} tick: {}".format(self.acceleration, self.tick)