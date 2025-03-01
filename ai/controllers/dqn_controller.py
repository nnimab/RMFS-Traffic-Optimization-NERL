from ai.traffic_controller import TrafficController
import random

class DQNController(TrafficController):
    """
    基於深度Q學習的交通控制器
    
    目前只是一個簡單的實現，用於測試框架
    使用固定的探索率隨機選擇方向
    """
    
    def __init__(self, exploration_rate=0.2, **kwargs):
        """
        初始化DQN控制器
        
        Args:
            exploration_rate (float): 探索率，控制隨機選擇動作的概率
            **kwargs: 其他參數
        """
        super().__init__(controller_name="DQN控制器")
        self.exploration_rate = exploration_rate
        self.last_direction = {}
        self.min_green_time = 10  # 最小綠燈時間
        self.last_change = {}  # 上次切換時間
    
    def get_direction(self, intersection, tick, warehouse):
        """
        使用簡化的DQN邏輯決定交通方向
        
        目前使用簡單的隨機策略，僅用於測試框架
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            str: 允許通行的方向，"Horizontal" 或 "Vertical"
        """
        intersection_id = intersection.id
        
        # 獲取當前方向，如果沒有則初始化
        if intersection_id not in self.last_direction:
            self.last_direction[intersection_id] = intersection.allowed_direction
            self.last_change[intersection_id] = tick
        
        current_dir = self.last_direction[intersection_id]
        last_change_tick = self.last_change[intersection_id]
        
        # 檢查是否滿足最小綠燈時間
        if tick - last_change_tick < self.min_green_time:
            return current_dir
            
        # 使用探索率隨機決定是否改變方向
        if random.random() < self.exploration_rate:
            # 隨機選擇新方向
            new_direction = random.choice(["Horizontal", "Vertical"])
            
            # 如果與當前方向不同，則更新狀態
            if new_direction != current_dir:
                self.last_direction[intersection_id] = new_direction
                self.last_change[intersection_id] = tick
            
            return new_direction
        else:
            # 否則保持當前方向
            return current_dir 