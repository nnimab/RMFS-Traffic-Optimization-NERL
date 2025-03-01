from datetime import datetime

class Landscape:
    def __init__(self, dimension):
        self.dimension = dimension
        self.total_objects = 0
        self.current_date_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.map = []
        self._objects = {}
        for i in range(self.dimension+1):
            one_row = []
            for j in range(self.dimension+1):
                one_row.append([])
            self.map.append(one_row)
    
    def getRobotObject(self):
        return self._objects
    
    def _setObjectNew(self, label, x, y, speed, acceleration, heading, state):
        self.total_objects += 1

        movement = 'vertical'
        if heading == 270 or heading == 90:
            movement = 'horizontal'

        self._objects[label] = {
            'label': label,
            'x': x,
            'y': y,
            'velocity': speed,
            'acceleration': acceleration,
            'heading': heading,
            'movement': movement,
            'state': state,
        }

        self.map[round(x)][round(y)].append(self._objects[label])

    def setObject(self, label, x, y, speed, acceleration, heading, state):
        if label not in self._objects:
            return self._setObjectNew(label, x, y, speed, acceleration, heading, state)
        
        old_x = round(self._objects[label]['x'])
        old_y = round(self._objects[label]['y'])
        
        # check if x or y has changed
        if round(x) != old_x or round(y) != old_y:
            # remove from old position
            to_iter = self.map[old_x][old_y] 
            for index, e in enumerate(to_iter):
                if e['label'] == label:
                    del to_iter[index]
                    break

            # add to new position
            self.map[round(x)][round(y)].append(self._objects[label])

        movement = 'vertical'
        if heading == 270 or heading == 90:
            movement = 'horizontal'

        self._objects[label] = {
            'label': label,
            'x': x,
            'y': y,
            'velocity': speed,
            'acceleration': acceleration,
            'heading': heading,
            'movement': movement,
            'state': state
        }

    def getNeighborObjectWithRadius(self, x, y, radius):
        i = x-radius
        j = y+radius
        check = 2*radius+1
        points_to_check = []
        result = []
        while i < x+check:
            j = y+radius
            while j > y-check:
                if i >= 0 and j >= 0:
                    if i != x or j != y:
                        points_to_check.append([i, j])
                j -= 1
            i += 1

        for p in points_to_check:
            s = self.map[p[0]][p[1]]
            if len(s) > 0:
                for obj in s:
                    result.append(self._objects[obj['label']])

        return result

    def getNeighborObject(self, x, y):
        s = self.map[round(x)][round(y)]
        if len(s) > 0:
            for obj in s:
                return self._objects[obj['label']]
        return None

    @property
    def objects(self):
        return self._objects

        