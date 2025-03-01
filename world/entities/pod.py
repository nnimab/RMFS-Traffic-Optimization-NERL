from typing import TYPE_CHECKING
from world.entities.object import Object
from lib.types.netlogo_coordinate import NetLogoCoordinate
if TYPE_CHECKING:
    from world.managers.pod_manager import PodManager

class Pod(Object):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, 'pod', x, y)
        self.pod_manager: PodManager = None
        self.pod_number = id
        self.shape = 'full square'
        self.skus = {}
        self.is_idle = True
        self.station = None
        self.need_replenishment = False

    def __eq__(self, other):
        if isinstance(other, Pod):
            return self.pod_number == other.pod_number
        return False

    def __hash__(self):
        return hash(self.pod_number)

    def __repr__(self):
        return f"Pod({self.pod_number})"

    def setPodManager(self, pod_manager):
        self.pod_manager = pod_manager

    def addSKU(self, sku, limit_qty, current_qty, threshold):
        """Add a new SKU with its limit, current quantity, and threshold."""
        self.skus[sku] = {
            'limit_qty': limit_qty,
            'current_qty': current_qty,
            'threshold': threshold
        }

    def isNeedReplenishment(self):
        """Check if 50% or more SKUs are below their threshold to determine if the pod needs to move to a
        replenishment station."""
        count_below_threshold = 0
        total_skus = len(self.skus)
        alpha = total_skus / 2
        for details in self.skus.values():
            # print(f"crt {details['current_qty']} limit {details['limit_qty']} th {details['threshold']}")
            if float(details['current_qty'])/float(details['limit_qty']) <= float(details['threshold']):
                count_below_threshold += 1

        if count_below_threshold >= alpha:
            return True
        return False

    def replenishAllSKU(self):
        """Replenish all SKUs by setting each SKU's current quantity to its limit quantity."""
        for sku in self.skus:
            self.skus[sku]['current_qty'] = self.skus[sku]['limit_qty']

    def pickSKU(self, sku, qty):
        self.skus[sku]['current_qty'] -= qty

    def getQuantity(self, sku):
        return self.skus[sku]['current_qty']

    def getUnassignedSKUs(self):
        """Return a list of SKUs that have not yet been assigned a pod."""
        unassigned_skus = [sku for sku, details in self.skus.items() if details['pod'] is None]
        return unassigned_skus

    def setPodStation(self, station):
        self.station = station
        return
    
    def removePodStation(self):
        self.station = None
        return

    def getAllSKUInPod(self):
        return self.skus