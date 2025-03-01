from lib.types.netlogo_coordinate import NetLogoCoordinate

class Job:
    def __init__(self, id: int, pod_coordinate: NetLogoCoordinate, station_id):
        self.id = id
        self.pod_coordinate = pod_coordinate
        self.pod = None
        self.station_id = station_id
        self.orders = []  # This will hold tuples of (order_id, sku, quantity)
        self.picking_delay_per_sku = 40 # Time for handling a task
        self.picking_delay = 0
        self.replenishment_delay_per_sku = 80
        self.replenishment_delay = 80
        self.is_finished = False

    def __str__(self):
        return f"Job: {self.id}, {self.pod_coordinate}, {self.station_id}, {self.orders}"

    def __repr__(self):
        return self.__str__()
    
    def addPickingTask(self, order_id, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order_id, sku, quantity))
        self.picking_delay += self.picking_delay_per_sku
    
    def addReplenishmentTask(self, pod):
        total_skus = len(pod.skus)
        self.replenishment_delay += total_skus * self.replenishment_delay_per_sku

    def isBeingProcessed(self):
        """Check if the job is being processed based on delays."""
        return self.picking_delay > 0 or self.replenishment_delay > 0

    def decrementDelay(self):
        """Decrement the picking or replenishment delay."""
        if self.picking_delay > 0:
            self.picking_delay -= 1
        elif self.replenishment_delay > 0:
            self.replenishment_delay -= 1

    def popOrder(self):
        return self.orders.pop(0)
    
