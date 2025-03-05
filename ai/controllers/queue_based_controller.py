from ai.traffic_controller import TrafficController

class QueueBasedController(TrafficController):
    """
    基於隊列的交通控制器
    
    根據路口垂直和水平方向的機器人數量和任務優先級來決定交通流向
    任務優先級：
    1. 最高優先級：送pod去撿貨站 (delivering_pod)
    2. 次高優先級：將pod送回倉庫 (returning_pod)
    3. 一般優先級：空車機器人去倉庫拿pod (taking_pod)
    """
    
    def __init__(self, min_green_time=1, bias_factor=1.5, **kwargs):
        """
        初始化基於隊列的控制器
        
        Args:
            min_green_time (int): 最小綠燈持續時間，避免頻繁切換造成的額外停車和能源消耗
            bias_factor (float): 方向偏好因子，用於調整水平和垂直方向的權重
            **kwargs: 其他參數
        """
        super().__init__(controller_name="隊列基控制器")
        self.min_green_time = min_green_time
        self.bias_factor = bias_factor
        # 定義任務優先級權重
        self.priority_weights = {
            "delivering_pod": 3.0,  # 送pod去撿貨站 (最高優先級)
            "returning_pod": 2.0,   # 將pod送回倉庫 (次高優先級)
            "taking_pod": 1.0,      # 空車機器人去倉庫拿pod (一般優先級)
            "idle": 0.5,            # 閒置狀態 (最低優先級)
            "station_processing": 0.0  # 在站台處理中的機器人不需要考慮
        }
    
    def get_direction(self, intersection, tick, warehouse):
        """
        根據路口兩個方向的機器人數量和任務優先級決定通行方向
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            str: 允許通行的方向，"Horizontal" 或 "Vertical"
        """
        # 檢查最小綠燈時間，避免頻繁切換
        if intersection.allowed_direction is not None and \
           intersection.durationSinceLastChange(tick) < self.min_green_time:
            return intersection.allowed_direction
        
        # 計算水平和垂直方向的加權優先級總和
        horizontal_priority = self._calculate_direction_priority(intersection.horizontal_robots)
        vertical_priority = self._calculate_direction_priority(intersection.vertical_robots)
        
        # 應用偏好因子 (水平方向通常有更多的機器人)
        horizontal_priority *= self.bias_factor
        
        # 打印調試信息
        print(f"Intersection {intersection.id} at ({intersection.pos_x}, {intersection.pos_y}):")
        print(f"  Horizontal priority: {horizontal_priority} (robots: {len(intersection.horizontal_robots)})")
        print(f"  Vertical priority: {vertical_priority} (robots: {len(intersection.vertical_robots)})")
        
        # 如果兩個方向都沒有機器人，保持當前狀態
        if len(intersection.horizontal_robots) == 0 and len(intersection.vertical_robots) == 0:
            return intersection.allowed_direction
        
        # 如果一個方向沒有機器人，另一個方向有，則選擇有機器人的方向
        if len(intersection.horizontal_robots) == 0:
            return "Vertical"
        if len(intersection.vertical_robots) == 0:
            return "Horizontal"
        
        # 根據優先級總和決定方向
        if horizontal_priority >= vertical_priority:
            return "Horizontal"
        else:
            return "Vertical"
    
    def _calculate_direction_priority(self, robots_dict):
        """
        計算特定方向的機器人優先級總和
        
        Args:
            robots_dict: 機器人字典，包含該方向的所有機器人
            
        Returns:
            float: 該方向的優先級總和
        """
        if not robots_dict:
            return 0
        
        total_priority = 0
        for robot in robots_dict.values():
            # 獲取機器人當前狀態的優先級權重
            weight = self.priority_weights.get(robot.current_state, 0.5)  # 默認權重為0.5
            total_priority += weight
            
        return total_priority
