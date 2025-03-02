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
        # 檢查坐標是否在有效範圍內
        new_x = round(x)
        new_y = round(y)
        if new_x < 0 or new_y < 0 or new_x > self.dimension or new_y > self.dimension:
            # 如果位置無效，不進行設置
            return
            
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

        self.map[new_x][new_y].append(self._objects[label])

    def setObject(self, label, x, y, speed, acceleration, heading, state):
        # 檢查新位置是否在有效範圍內
        new_x = round(x)
        new_y = round(y)
        if new_x < 0 or new_y < 0 or new_x > self.dimension or new_y > self.dimension:
            # 如果新位置無效，不進行更新
            return
            
        if label not in self._objects:
            return self._setObjectNew(label, x, y, speed, acceleration, heading, state)
        
        old_x = round(self._objects[label]['x'])
        old_y = round(self._objects[label]['y'])
        
        # 檢查舊位置是否在有效範圍內
        if old_x >= 0 and old_y >= 0 and old_x <= self.dimension and old_y <= self.dimension:
            # check if x or y has changed
            if new_x != old_x or new_y != old_y:
                # remove from old position
                to_iter = self.map[old_x][old_y] 
                for index, e in enumerate(to_iter):
                    if e['label'] == label:
                        del to_iter[index]
                        break

                # add to new position
                self.map[new_x][new_y].append(self._objects[label])
        else:
            # 如果舊位置無效，則只添加到新位置
            self.map[new_x][new_y].append(self._objects[label])

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
                if i >= 0 and j >= 0 and i < self.dimension+1 and j < self.dimension+1:
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
        # 添加邊界檢查，確保坐標在有效範圍內
        x_rounded = round(x)
        y_rounded = round(y)
        if x_rounded < 0 or y_rounded < 0 or x_rounded > self.dimension or y_rounded > self.dimension:
            return None
            
        s = self.map[x_rounded][y_rounded]
        if len(s) > 0:
            for obj in s:
                return self._objects[obj['label']]
        return None

    @property
    def objects(self):
        return self._objects

        