import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def astar(grid, start, end):
    def heuristic(node, end):
        return abs(node.x - end.x) + abs(node.y - end.y)

    open_set = []
    closed_set = set()

    start_node = Node(start[0], start[1])
    end_node = Node(end[0], end[1])

    open_set.append(start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.x == end_node.x and current_node.y == end_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = current_node.x + dx, current_node.y + dy

            if (
                0 <= x < len(grid)
                and 0 <= y < len(grid[0])
                and grid[x][y] == 0
                and (x, y) not in closed_set
            ):
                neighbor = Node(x, y, current_node)
                neighbor.g = current_node.g + 1
                neighbor.h = heuristic(neighbor, end_node)

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)

    return None

class NetLogoCoordinate:
    x = 0
    y = 0
    dimension = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "({},{})".format(self.x, self.y)


class Coordinate:
    x = 0
    y = 0
    landscape = None

    def __init__(self, x=0, y=0, landscape=None):
        self.x = x
        self.y = y
        self.landscape = landscape

class Landscape:
    dimension = 0
    _map = []

    def __init__(self, dimension):
        self.dimension = dimension
        for i in range(self.dimension+1):
            one_row = []
            for j in range(self.dimension+1):
                one_row.append(0)
            self._map.append(one_row)
    
    def cloneMap(self):
        return self._map.copy()
    
    def printMap(self, to_print=None):
        if to_print is None:
            to_print = self._map
            
        idx = 0
        for i in to_print:
            print(str(idx) + " ", end="")
            print(i)
            idx += 1
    
pods = [
    [12, 2],
    [13, 2],
    [14, 2],
    [15, 2],
    [16, 2],
    [12, 3],
    [13, 3],
    [14, 3],
    [15, 3],
    [16, 3],
    [18, 2],
    [19, 2],
    [20, 2],
    [21, 2],
    [22, 2],
    [18, 3],
    [19, 3],
    [20, 3],
    [21, 3],
    [22, 3],
    [12, 5],
    [13, 5],
    [14, 5],
    [15, 5],
    [16, 5],
    [12, 6],
    [13, 6],
    [14, 6],
    [15, 6],
    [16, 6],
    [18, 5],
    [19, 5],
    [20, 5],
    [21, 5],
    [22, 5],
    [18, 6],
    [19, 6],
    [20, 6],
    [21, 6],
    [22, 6],
]

landscape = Landscape(52)
for p in pods:
    coor = NetLogoCoordinate(p[0], p[1])
    landscape.setObject(coor, 1)

_from = NetLogoCoordinate(23, 6)
_to = NetLogoCoordinate(12, 2)

# # Example usage:
# grid = [
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0],
# ]

# start = (0, 0)
# end = (4, 3)

# for r in grid:
#     print(r)
# print("====")
# path = astar(grid, start, end)

# if path:
#     for x, y in path:
#         grid[x][y] = 2

#     for row in grid:
#         print(row)
# else:
#     print("No path found.")

