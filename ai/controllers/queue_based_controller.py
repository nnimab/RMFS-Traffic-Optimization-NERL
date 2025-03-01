from ai.traffic_controller import TrafficController

class QueueBasedController(TrafficController):
    """
    基於隊列長度的交通控制器（啟發式方法）
    
    根據等待機器人數量調整交通方向
    為水平方向設置偏好因子，反映pods分佈特性
    設置最小綠燈時間，避免頻繁切換
    """
    
    def __init__(self, min_green_time=10, bias_factor=1.5, **kwargs):
        """
        初始化基於隊列的控制器
        
        Args:
            min_green_time (int): 最小綠燈時間，避免頻繁切換
            bias_factor (float): 水平方向偏好因子，反映pods分佈特性
            **kwargs: 其他參數
        """
        super().__init__(controller_name="隊列基控制器")
        self.min_green_time = min_green_time  # 最小綠燈時間
        self.bias_factor = bias_factor  # 水平方向偏好因子
        self.last_change = {}  # 記錄每個交叉路口上次更改的時間
        self.current_direction = {}  # 當前方向
    
    def get_direction(self, intersection, tick, warehouse):
        """
        根據交叉路口隊列長度確定交通方向
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            str: 允許通行的方向，"Horizontal" 或 "Vertical"
        """
        intersection_id = intersection.id
        
        # 獲取水平和垂直方向的隊列長度
        horizontal_count = len(intersection.horizontal_robots)
        vertical_count = len(intersection.vertical_robots)
        
        # 考慮方向偏好 - 水平方向有優勢
        horizontal_count = horizontal_count * self.bias_factor
        
        # 獲取當前方向，如果沒有則設為None
        if intersection_id not in self.current_direction:
            self.current_direction[intersection_id] = intersection.allowed_direction
            self.last_change[intersection_id] = tick
        
        current_dir = self.current_direction[intersection_id]
        last_change_tick = self.last_change[intersection_id]
        
        # 檢查是否滿足最小綠燈時間
        if tick - last_change_tick < self.min_green_time:
            return current_dir
        
        # 根據隊列長度決定方向
        if horizontal_count > vertical_count and current_dir != "Horizontal":
            self.current_direction[intersection_id] = "Horizontal"
            self.last_change[intersection_id] = tick
            return "Horizontal"
        elif vertical_count > horizontal_count and current_dir != "Vertical":
            self.current_direction[intersection_id] = "Vertical"
            self.last_change[intersection_id] = tick
            return "Vertical"
        
        # 如果隊列長度相同，保持當前方向
        return current_dir 