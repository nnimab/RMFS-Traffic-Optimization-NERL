from ai.traffic_controller import TrafficController
import random
import numpy as np
import torch
from ai.deep_q_network import DeepQNetwork
import os
from world.entities.robot import Robot

class DQNController(TrafficController):
    """
    基於深度Q學習的交通控制器
    
    使用強化學習優化交通信號控制，考慮多種特徵如等待車輛數量、
    能源消耗、停止啟動次數等，隨時間自適應學習最優策略。
    """
    
    def __init__(self, 
                 exploration_rate=0.2, 
                 state_size=8, 
                 action_size=2, 
                 batch_size=32, 
                 update_target_every=100, 
                 training_interval=5,
                 min_green_time=10,
                 **kwargs):
        """
        初始化DQN控制器
        
        Args:
            exploration_rate (float): 探索率，控制隨機選擇動作的概率
            state_size (int): 狀態空間維度
            action_size (int): 動作空間維度
            batch_size (int): 訓練批次大小
            update_target_every (int): 每多少時間步更新目標網絡
            training_interval (int): 每多少時間步進行一次訓練
            min_green_time (int): 最小綠燈時間
            **kwargs: 其他參數
        """
        super().__init__(controller_name="DQN控制器")
        
        # DQN控制器參數
        self.exploration_rate = exploration_rate
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.training_interval = training_interval
        self.min_green_time = min_green_time
        
        # 交叉路口狀態追踪
        self.last_direction = {}
        self.last_change_time = {}
        self.current_green_time = {}
        self.step_counter = 0
        
        # 添加狀態和動作記錄初始化
        self.last_state = {}
        self.last_action = {}
        
        # DQN網絡
        self.dqn = DeepQNetwork(state_size, action_size)
        
        # 自動保存設置
        self.save_at_ticks = [5000, 10000, 20000]
        self.last_saved_tick = 0
        
        # 確保models目錄存在
        os.makedirs("models", exist_ok=True)
    
    def get_state(self, intersection, tick, warehouse):
        """
        獲取交叉路口的狀態特徵
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            np.array: 狀態特徵向量
        """
        # 當前允許方向的編碼 (0: Horizontal, 1: Vertical)
        current_direction = 0 if intersection.allowed_direction == "Horizontal" else 1
        
        # 當前方向已持續的時間
        intersection_id = intersection.id
        direction_duration = 0
        if intersection_id in self.last_change_time:
            direction_duration = tick - self.last_change_time[intersection_id]
        
        # 規範化持續時間 (除以100做規範化)
        normalized_duration = direction_duration / 100.0
        
        # 水平和垂直方向等待的機器人數量
        horizontal_count = len(intersection.horizontal_robots)
        vertical_count = len(intersection.vertical_robots)
        
        # 規範化機器人數量 (除以10做規範化)
        normalized_horizontal = horizontal_count / 10.0
        normalized_vertical = vertical_count / 10.0
        
        # 水平和垂直方向的機器人最長等待時間
        max_horizontal_wait = 0
        max_vertical_wait = 0
        
        if horizontal_count > 0:
            # 使用 current_intersection_start_time 替代 last_move_tick
            wait_times = []
            for robot in intersection.horizontal_robots.values():
                if robot.current_intersection_start_time is not None:
                    wait_times.append(tick - robot.current_intersection_start_time)
                else:
                    wait_times.append(0)
            max_horizontal_wait = max(wait_times) if wait_times else 0
        
        if vertical_count > 0:
            # 使用 current_intersection_start_time 替代 last_move_tick
            wait_times = []
            for robot in intersection.vertical_robots.values():
                if robot.current_intersection_start_time is not None:
                    wait_times.append(tick - robot.current_intersection_start_time)
                else:
                    wait_times.append(0)
            max_vertical_wait = max(wait_times) if wait_times else 0
        
        # 規範化最長等待時間 (除以100做規範化)
        normalized_h_wait = max_horizontal_wait / 100.0
        normalized_v_wait = max_vertical_wait / 100.0
        
        # 構建特徵向量
        state = np.array([
            current_direction,
            normalized_duration,
            normalized_horizontal,
            normalized_vertical,
            normalized_h_wait,
            normalized_v_wait,
            0.0,  # 原本是 intersection.current_intersection_energy_consumption / 1000.0，現在使用預設值 0.0
            0.0   # 原本是 intersection.current_intersection_stop_and_go / 10.0，現在使用預設值 0.0
        ])
        
        return state
    
    def get_reward(self, intersection, warehouse, state, next_state):
        """
        計算獎勵函數
        
        Args:
            intersection: 交叉路口對象
            warehouse: 倉庫對象
            state: 當前狀態
            next_state: 下一個狀態
            
        Returns:
            float: 獎勵值
        """
        # 計算或取得相關指標
        # 原本的屬性不存在，使用計算的方式或預設值
        energy_consumption = 0.0  # 原本是 intersection.current_intersection_energy_consumption
        
        # 使用交叉路口的停止和啟動統計數據
        horizontal_stop_go = intersection.calculateAverageStopAndGo('horizontal')
        vertical_stop_go = intersection.calculateAverageStopAndGo('vertical')
        stop_and_go = horizontal_stop_go + vertical_stop_go  # 兩個方向的平均值
        
        # 計算水平和垂直方向的機器人通過數量變化
        prev_horizontal = state[2] * 10
        prev_vertical = state[3] * 10
        next_horizontal = next_state[2] * 10
        next_vertical = next_state[3] * 10
        
        passed_horizontal = max(0, prev_horizontal - next_horizontal)
        passed_vertical = max(0, prev_vertical - next_vertical)
        total_passed = passed_horizontal + passed_vertical
        
        # 定義獎勵組件
        throughput_reward = total_passed * 2.0  # 通過的機器人數量獎勵
        energy_penalty = -energy_consumption * 0.01  # 能源消耗懲罰
        stop_go_penalty = -stop_and_go * 0.05  # 停止啟動懲罰
        
        # 綜合獎勵
        reward = throughput_reward + energy_penalty + stop_go_penalty
        
        return reward
    
    def get_direction(self, intersection, tick, warehouse):
        """
        根據當前路況決定交通信號方向
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時刻
            warehouse: 倉庫對象
        
        Returns:
            int: 新的方向（0：東西向，1：南北向）
        """
        intersection_id = intersection.id
        
        # 初始化交叉路口狀態
        if intersection_id not in self.last_direction:
            self.last_direction[intersection_id] = random.randint(0, 1)
            self.last_change_time[intersection_id] = 0
            self.current_green_time[intersection_id] = 0
            
            # 初始化DQN網絡（如果尚未初始化）
            if not hasattr(self, 'dqn'):
                self.dqn = DeepQNetwork(self.state_size, self.action_size)
        
        current_dir = self.last_direction[intersection_id]
        last_change_tick = self.last_change_time[intersection_id]
        
        # 檢查是否滿足最小綠燈時間
        if tick - last_change_tick < self.min_green_time:
            return current_dir
        
        # 獲取當前狀態
        current_state = self.get_state(intersection, tick, warehouse)
        
        # 執行動作（決策）
        action = self.dqn.act(current_state)
        new_direction = "Horizontal" if action == 0 else "Vertical"
        
        # 如果這個交叉路口之前有狀態和動作記錄，則計算獎勵並存儲經驗
        if intersection_id in self.last_state and intersection_id in self.last_action:
            prev_state = self.last_state[intersection_id]
            prev_action = self.last_action[intersection_id]
            
            # 計算獎勵
            reward = self.get_reward(intersection, warehouse, prev_state, current_state)
            
            # 判斷是否為終止狀態（這裡簡單設為False，因為交通控制是連續任務）
            done = False
            
            # 存儲經驗
            self.dqn.remember(prev_state, prev_action, reward, current_state, done)
        
        # 更新狀態和動作記錄
        self.last_state[intersection_id] = current_state
        self.last_action[intersection_id] = action
        
        # 如果與當前方向不同，則更新狀態
        if new_direction != current_dir:
            self.last_direction[intersection_id] = new_direction
            self.last_change_time[intersection_id] = tick
        
        # 計數器增加
        self.step_counter += 1
        
        # 定期訓練網絡
        if self.step_counter % self.training_interval == 0:
            if Robot.DEBUG_LEVEL > 0:
                print(f"[Training] Starting DQN training at step {self.step_counter}, tick {tick}")
            self.dqn.replay(self.batch_size)
        
        # 定期更新目標網絡
        if self.step_counter % self.update_target_every == 0:
            if Robot.DEBUG_LEVEL > 0:
                print(f"[Training] Updating target network at step {self.step_counter}")
            self.dqn.update_target_model()
        
        # 定期保存模型
        self.step_counter += 1
        
        # 每1000步常規保存
        if self.step_counter % 1000 == 0:
            self.dqn.save_model(tick=tick)
            
        # 在特定ticks保存模型里程碑
        if tick in self.save_at_ticks and tick > self.last_saved_tick:
            milestone_name = f"dqn_traffic_milestone_{tick}"
            self.dqn.save_model(model_name=milestone_name)
            print(f"Milestone model saved at tick {tick}")
            self.last_saved_tick = tick
        
        return new_direction
    
    def save_model(self, tick=None, model_name=None):
        """
        保存DQN模型
        
        Args:
            tick (int, optional): 當前時間刻，用於檔案命名
            model_name (str, optional): 自定義模型名稱
        """
        self.dqn.save_model(tick=tick, model_name=model_name)
    
    def load_model(self, model_path=None, tick=None):
        """
        加載DQN模型
        
        Args:
            model_path (str, optional): 完整模型路徑
            tick (int, optional): 特定時間點的模型(例如：5000)
        
        Returns:
            bool: 是否成功加載模型
        """
        # 如果指定了tick，優先使用milestone模型
        if tick is not None and not model_path:
            model_path = f"models/dqn_traffic_milestone_{tick}.pth"
            
        return self.dqn.load_model(model_path) 