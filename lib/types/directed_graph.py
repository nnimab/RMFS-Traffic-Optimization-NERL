import networkx as nx

from lib.types.netlogo_coordinate import NetLogoCoordinate
class DirectedGraph:
    key = ''

    def __init__(self):
        """Initialize an instance with a directed graph."""
        self.graph = nx.DiGraph()

    @staticmethod
    def nodeValid(node):
        """Check if a node is valid based on custom logic.

        Args:
            node (str): The node in format 'x,y'.

        Returns:
            bool: True if the node is valid, False otherwise.
        """
        x, y = map(int, node.split(","))
        return x >= 2 and y >= 0

    def addNode(self, node):
        """Add a node to the graph if it's valid.

        Args:
            node (str): The node to add.
        """
        if self.nodeValid(node):
            self.graph.add_node(node)

    def addEdge(self, start, end, weight):
        """Add an edge between two nodes with a weight if both nodes are valid.

        Args:
            start (str): The start node.
            end (str): The end node.
            weight (float): The weight of the edge.
        """
        if self.nodeValid(start) and self.nodeValid(end):
            self.graph.add_edge(start, end, weight=weight)
    
    def add_all_direction_paths(self, obj_key, weight):
        x, y = map(int, obj_key.split(','))
        directions = {
            'left': (x - 1, y),
            'right': (x + 1, y),
            'up': (x, y + 1),
            'down': (x, y - 1)
        }

        for (key, tuple) in directions.items():
            (nx, ny) = tuple
            neighbor_key = f"{nx},{ny}"
            self.addEdge(obj_key, neighbor_key, weight=weight)

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

    def dijkstraModified(self, start, end, penalties, zone_boundary, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Create a copy of the graph so we can modify it without affecting the original
        G = self.graph.copy()

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 10000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 10000

        # Increase the weight of edges in every zone based on the penalty
        for index, zone in enumerate(zone_boundary):
            for row in range(zone[1][0], zone[0][0]):
                for col in range(zone[0][1], zone[1][1]):
                    coordinate_str = f"{row},{col}"
                    for neighbor in list(G.neighbors(coordinate_str)) + list(G.predecessors(coordinate_str)):
                        if G.has_edge(neighbor, coordinate_str):
                            G[neighbor][coordinate_str]['weight'] = penalties[index]
                        if G.has_edge(coordinate_str, neighbor):
                            G[coordinate_str][neighbor]['weight'] = penalties[index]

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except nx.NetworkXNoPath:
            return None

    def dijkstra(self, start, end, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Create a copy of the graph so we can modify it without affecting the original
        G = self.graph.copy()

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 1000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 1000

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except nx.NetworkXNoPath:
            return None