from typing import TYPE_CHECKING
from world.entities.object import Object
from lib.enum.area_path_type import AreaPathType
if TYPE_CHECKING:
    from world.managers.area_path_manager import AreaPathManager

class AreaPath(Object):
    def __init__(self, id: int, x: int, y: int, type_value: AreaPathType):
        super().__init__(id, 'area_path', x, y)
        self.area_path_manager: AreaPathManager = None
        self.type_value = type_value
        self.shape = self.setShape()
    
    def setAreaPathManager(self, area_path_manager):
        self.area_path_manager = area_path_manager

    def setShape(self):
        if (
            self.type_value == AreaPathType.Empty or 
            self.type_value == AreaPathType.Intersection or
            self.type_value == AreaPathType.Void
        ):
            return 'empty-space'
        elif self.type_value == AreaPathType.Pod:
            return 'full square'
        elif self.type_value == AreaPathType.RechargeStation:
            return 'square 2'
        elif self.type_value == AreaPathType.LeftDirection:
            return 'arrow-left'
        elif self.type_value == AreaPathType.RightDirection:
            return 'arrow-right'
        elif self.type_value == AreaPathType.UpDirection:
            return 'arrow-up'
        elif self.type_value == AreaPathType.DownDirection:
            return 'arrow-down'
        elif (
            self.type_value == AreaPathType.PickerStaff or 
            self.type_value == AreaPathType.ReplenishmentStaff
        ):
            return 'person-red'
        elif (
            self.type_value == AreaPathType.PickerMoveLeftPath or 
            self.type_value == AreaPathType.PickerMoveRightPath or 
            self.type_value == AreaPathType.ReplenishmentMoveRightPath or 
            self.type_value == AreaPathType.ReplenishmentMoveLeftPath
        ):
            return 'rail'
        elif (
            self.type_value == AreaPathType.PickerMoveDownPath or 
            self.type_value == AreaPathType.ReplenishmentMoveDownPath
        ):
            return 'rail-triangle'
        elif (
            self.type_value == AreaPathType.PickerTurnLeftDownPath or 
            self.type_value == AreaPathType.PickerTurnDownRightPath or
            self.type_value == AreaPathType.PickerTurnRightDownPath or
            self.type_value == AreaPathType.PickerTurnDownLeftPath or
            self.type_value == AreaPathType.ReplenishmentTurnRightDownPath or 
            self.type_value == AreaPathType.ReplenishmentTurnDownLeftPath or 
            self.type_value == AreaPathType.ReplenishmentTurnLeftDownPath or 
            self.type_value == AreaPathType.ReplenishmentTurnDownRightPath
        ):
            return 'rail-corner'
        else:
            return 'empty-space'
