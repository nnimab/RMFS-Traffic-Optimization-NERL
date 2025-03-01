from abc import ABC, abstractmethod

class TrafficController(ABC):
    """
    交通控制器基類
    
    這是所有交通控制策略的抽象基類，定義了統一的介面
    具體的控制策略需要繼承此類並實現其方法
    """
    
    def __init__(self, controller_name):
        """
        初始化交通控制器
        
        Args:
            controller_name (str): 控制器名稱，用於識別和記錄
        """
        self.controller_name = controller_name
        self.statistics = {
            "direction_changes": 0,
            "total_wait_time": 0,
            "total_stop_and_go": 0,
            "total_energy": 0
        }
    
    @abstractmethod
    def get_direction(self, intersection, tick, warehouse):
        """
        根據交叉路口狀態確定應該允許通行的方向
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象，提供更多系統信息
            
        Returns:
            str: 允許通行的方向，可能的值包括 "Horizontal"、"Vertical" 或 None
        """
        pass
    
    def update_statistics(self, stat_type, value):
        """
        更新控制器統計信息
        
        Args:
            stat_type (str): 統計類型
            value: 要更新的值
        """
        if stat_type in self.statistics:
            self.statistics[stat_type] += value
    
    def get_statistics(self):
        """
        獲取控制器統計信息
        
        Returns:
            dict: 控制器統計信息
        """
        return self.statistics
    
    def reset_statistics(self):
        """
        重置控制器統計信息
        """
        for key in self.statistics:
            self.statistics[key] = 0


class TrafficControllerFactory:
    """
    交通控制器工廠類
    
    用於創建不同類型的交通控制器實例
    """
    
    @staticmethod
    def create_controller(controller_type, **kwargs):
        """
        創建指定類型的交通控制器
        
        Args:
            controller_type (str): 控制器類型，可能的值包括 "time_based"、"queue_based"、"dqn" 和 "nerl"
            **kwargs: 傳遞給控制器構造函數的其他參數
            
        Returns:
            TrafficController: 創建的交通控制器實例
            
        Raises:
            ValueError: 如果指定的控制器類型無效
        """
        if controller_type == "time_based":
            from ai.controllers.time_based_controller import TimeBasedController
            return TimeBasedController(**kwargs)
        elif controller_type == "queue_based":
            from ai.controllers.queue_based_controller import QueueBasedController
            return QueueBasedController(**kwargs)
        elif controller_type == "dqn":
            from ai.controllers.dqn_controller import DQNController
            return DQNController(**kwargs)
        elif controller_type == "nerl":
            from ai.controllers.nerl_controller import NEController
            return NEController(**kwargs)
        else:
            raise ValueError(f"無效的控制器類型: {controller_type}") 