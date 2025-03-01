from enum import Enum

class AreaPathType(Enum):
    Empty = 0
    Pod = 1
    RechargeStation = 2
    Intersection = 3
    LeftDirection = 4
    RightDirection = 5
    UpDirection = 6
    DownDirection = 7
    PickerStaff = 11
    PickerMoveLeftPath = 12
    PickerMoveRightPath = 13
    PickerMoveDownPath = 14
    PickerTurnLeftDownPath = 16
    PickerTurnDownRightPath = 17
    PickerTurnRightDownPath = 18
    PickerTurnDownLeftPath = 19
    ReplenishmentStaff = 21
    ReplenishmentMoveRightPath = 22
    ReplenishmentMoveLeftPath = 23
    ReplenishmentMoveDownPath = 24
    ReplenishmentTurnRightDownPath = 26
    ReplenishmentTurnDownLeftPath = 27
    ReplenishmentTurnLeftDownPath = 28
    ReplenishmentTurnDownRightPath = 29
    Void = 99
