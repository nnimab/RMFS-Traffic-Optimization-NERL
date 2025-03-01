class Heading:
    heading = 0

    def __init__(self, heading=0):
        self.heading = heading
    
    def __repr__(self):
        return "heading: {}".format(self.heading)
    
    def getHeading(self):
        return self.heading