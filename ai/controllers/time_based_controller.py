from ai.traffic_controller import TrafficController

class TimeBasedController(TrafficController):
    """
    基於時間的交通控制器
    
    使用固定的時間週期來切換交叉路口的通行方向
    水平方向（左右）因pods數量較多，設置更長的綠燈時間
    不考慮實時交通狀況
    """
    
    def __init__(self, horizontal_green_time=70, vertical_green_time=30, **kwargs):
        """
        初始化基於時間的控制器
        
        Args:
            horizontal_green_time (int): 水平方向綠燈持續時間
            vertical_green_time (int): 垂直方向綠燈持續時間
            **kwargs: 其他參數
        """
        super().__init__(controller_name="時間基控制器")
        self.horizontal_green_time = horizontal_green_time  # 水平方向綠燈時間更長
        self.vertical_green_time = vertical_green_time
        self.cycle_length = horizontal_green_time + vertical_green_time
    
    def get_direction(self, intersection, tick, warehouse):
        """
        根據當前時間週期確定交通方向
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            str: 允許通行的方向，"Horizontal" 或 "Vertical"
        """
        # 計算當前週期內的時間位置
        cycle_position = tick % self.cycle_length
        
        # 水平方向有更長的綠燈時間
        if cycle_position < self.horizontal_green_time:
            return "Horizontal"
        else:
            return "Vertical" 