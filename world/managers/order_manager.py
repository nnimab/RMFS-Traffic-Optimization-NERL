from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING
from world.entities.order import Order
if TYPE_CHECKING:
    from world.warehouse import Warehouse

class OrderManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.orders: List[Order] = []
        self.order_id_to_order: Dict[int, Order] = {}
        self.finished_orders: List[Order] = []
        self.unfinished_orders: List[Order] = []

    def createOrder(self, order_id, order_arrival: int):
        new_order = Order(order_id, order_arrival)
        self.orders.append(new_order)
        self.order_id_to_order[new_order.id] = new_order
        self.unfinished_orders.append(new_order)
        return new_order

    def getOrderById(self, order_id) -> Optional[Order]:
        """Retrieve an order by its ID using the dictionary for quick access."""
        return self.order_id_to_order.get(order_id, None)

    def removeOrder(self, order:Order):
        self.orders.remove(order)

    def finishOrder(self, order_id, tick: int):
        """Move an order from the unfinished_orders list to the finished_orders list."""
        order = self.getOrderById(order_id)
        order.completeOrder(tick)
        if order and order in self.unfinished_orders:
            self.unfinished_orders.remove(order)
            self.finished_orders.append(order)
