class Order:
    def __init__(self, order_id, order_arrival):
        self.id = order_id
        self.order_arrival = order_arrival
        self.process_start_time = -1
        self.order_complete_time = -1
        self.station_id = None
        self.skus = {}
        self.status = -3

    def __str__(self):
        return f"Order(order_id={self.id}, order_arrival={self.order_arrival}, process_start_time={self.process_start_time}, order_complete_time={self.order_complete_time}, station_id={self.station_id}, skus={self.skus})"

    def __repr__(self):
        return self.__str__()

    def assignStation(self, station_id):
        self.station_id = station_id

    def addSKU(self, sku, total_quantity):
        self.skus[sku] = {
            'total_quantity': total_quantity,
            'quantity_committed': 0,
            'quantity_delivered': 0
        }

    def hasSKU(self, sku):
        if sku in self.skus:
            return True
        return False

    def commitQuantity(self, sku, quantity):
        self.skus[sku]['quantity_committed'] += quantity

    def deliverQuantity(self, sku, quantity):
        self.skus[sku]['quantity_delivered'] += quantity
        self.skus[sku]['quantity_committed'] -= quantity

    def getRemainingSKU(self):
        """Return a dictionary of SKUs with their remaining quantities to be fulfilled."""
        remaining_skus = {
            sku: details['total_quantity'] - (details['quantity_delivered'] + details['quantity_committed'])
            for sku, details in self.skus.items()
            if details['total_quantity'] > (details['quantity_delivered'] + details['quantity_committed'])
        }
        return remaining_skus

    def startProcessing(self, start_time):
        """Record the start time for order processing."""
        self.process_start_time = start_time

    def completeOrder(self, complete_time):
        """Record the time when order processing is completed."""
        self.order_complete_time = complete_time

    def getQuantityLeftForSKU(self, sku):
        """Return the total quantity left to be delivered for the specified SKU, including committed quantities."""
        details = self.skus[sku]
        remaining = details['total_quantity'] - (details['quantity_delivered'] + details['quantity_committed'])
        return remaining

    def isOrderCompleted(self):
        """Check if all SKUs in the order have been delivered as per the total quantity."""
        return all(details['total_quantity'] == details['quantity_delivered'] for details in self.skus.values())

    def getProcessingTime(self):
        """Calculate and return the total processing time from start to completion, if available."""
        return self.order_complete_time - self.process_start_time
