from ai.traffic_controller import TrafficController
from ai.deep_q_network import DeepQNetwork
import numpy as np
import torch


class DQNController(TrafficController):
    """
    基於深度強化學習的交通控制器
    
    結合了基於隊列控制器的規則邏輯與深度Q網絡的強化學習能力
    可以適應不同交通模式並優化控制決策
    """
    
    def __init__(self, min_green_time=1, bias_factor=1.5, state_size=8, action_size=3, 
                 max_wait_threshold=50, model_name="dqn_traffic", **kwargs):
        """
        初始化DQN控制器
        
        Args:
            min_green_time (int): 最小綠燈持續時間，避免頻繁切換
            bias_factor (float): 方向偏好因子，調整水平和垂直方向的權重
            state_size (int): 狀態空間維度
            action_size (int): 動作空間維度
            max_wait_threshold (int): 機器人最大等待時間閾值，用於防鎖死
            model_name (str): 模型名稱，用於保存和加載模型
            **kwargs: 其他參數
        """
        super().__init__(controller_name="DQN控制器")
        self.min_green_time = min_green_time
        self.bias_factor = bias_factor
        self.state_size = state_size
        self.action_size = action_size
        self.max_wait_threshold = max_wait_threshold
        self.model_name = model_name
        
        # 定義任務優先級權重 (從隊列控制器繼承)
        self.priority_weights = {
            "delivering_pod": 3.0,  # 送pod去撿貨站 (最高優先級)
            "returning_pod": 2.0,   # 將pod送回倉庫 (次高優先級)
            "taking_pod": 1.0,      # 空車機器人去倉庫拿pod (一般優先級)
            "idle": 0.5,            # 閒置狀態 (最低優先級)
            "station_processing": 0.0  # 在站台處理中的機器人不需要考慮
        }
        
        # 初始化深度Q網絡
        self.dqn = DeepQNetwork(state_size=state_size, action_size=action_size, model_name=model_name)
        
        # 用於存儲每個交叉路口的先前狀態和動作
        self.previous_states = {}
        self.previous_actions = {}
        
        # 是否處於訓練模式
        self.is_training = True
    
    def get_state(self, intersection, tick, warehouse):
        """
        獲取交叉路口的當前狀態向量
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            numpy.ndarray: 表示當前狀態的向量
        """
        # 當前允許方向編碼
        dir_code = 0
        if intersection.allowed_direction == "Vertical":
            dir_code = 1
        elif intersection.allowed_direction == "Horizontal":
            dir_code = 2
        
        # 自上次信號變化以來的時間
        time_since_change = intersection.durationSinceLastChange(tick)
        
        # 各方向機器人數量
        h_count = len(intersection.horizontal_robots)
        v_count = len(intersection.vertical_robots)
        
        # 優先級機器人數量 (delivering_pod 狀態)
        h_priority = len([robot for robot in intersection.horizontal_robots.values() 
                          if robot.current_state == "delivering_pod"])
        v_priority = len([robot for robot in intersection.vertical_robots.values() 
                          if robot.current_state == "delivering_pod"])
        
        # 計算平均等待時間
        h_wait_time, v_wait_time = intersection.calculateAverageWaitingTimePerDirection(tick)
        
        # 歸一化處理
        time_norm = min(time_since_change / 20.0, 1.0)  # 假設20個tick是最大值
        h_count_norm = min(h_count / 10.0, 1.0)  # 假設最多10個機器人
        v_count_norm = min(v_count / 10.0, 1.0)
        h_wait_norm = min(h_wait_time / 50.0, 1.0)  # 假設最長等待時間為50
        v_wait_norm = min(v_wait_time / 50.0, 1.0)
        
        state = [
            dir_code / 2.0,  # 歸一化到[0,1]
            time_norm,
            h_count_norm,
            v_count_norm,
            h_priority / max(h_count, 1),  # 優先機器人比例
            v_priority / max(v_count, 1),
            h_wait_norm,
            v_wait_norm
        ]
        
        return np.array(state)
    
    def get_direction(self, intersection, tick, warehouse):
        """
        根據當前狀態決定交通方向
        
        結合防鎖死機制、最小綠燈時間約束和DQN決策
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            str: 允許通行的方向 "Horizontal" 或 "Vertical"
        """
        # 防鎖死機制檢查 - 計算每個方向的最大等待時間
        max_wait_time_h = 0
        max_wait_time_v = 0
        
        for robot in intersection.horizontal_robots.values():
            if robot.current_intersection_start_time is not None:
                wait_time = tick - robot.current_intersection_start_time
                max_wait_time_h = max(max_wait_time_h, wait_time)
        
        for robot in intersection.vertical_robots.values():
            if robot.current_intersection_start_time is not None:
                wait_time = tick - robot.current_intersection_start_time
                max_wait_time_v = max(max_wait_time_v, wait_time)
        
        # 如果有機器人等待時間超過閾值，優先讓其通行
        if max_wait_time_h > self.max_wait_threshold:
            print(f"Intersection {intersection.id}: Emergency direction change to Horizontal due to long wait time ({max_wait_time_h} ticks)")
            return "Horizontal"
        
        if max_wait_time_v > self.max_wait_threshold:
            print(f"Intersection {intersection.id}: Emergency direction change to Vertical due to long wait time ({max_wait_time_v} ticks)")
            return "Vertical"
        
        # 檢查最小綠燈時間，避免頻繁切換
        if intersection.allowed_direction is not None and \
           intersection.durationSinceLastChange(tick) < self.min_green_time:
            return intersection.allowed_direction
        
        # 如果兩個方向都沒有機器人，保持當前狀態
        if len(intersection.horizontal_robots) == 0 and len(intersection.vertical_robots) == 0:
            return intersection.allowed_direction
        
        # 如果一個方向沒有機器人，另一個方向有，則選擇有機器人的方向
        if len(intersection.horizontal_robots) == 0:
            return "Vertical"
        if len(intersection.vertical_robots) == 0:
            return "Horizontal"
        
        # 使用DQN選擇動作
        state = self.get_state(intersection, tick, warehouse)
        
        # 保存當前狀態用於後續學習
        if self.is_training:
            intersection_id = intersection.id
            self.previous_states[intersection_id] = state
            
            action = self.dqn.act(state)
            self.previous_actions[intersection_id] = action
        else:
            # 推理模式 - 只選擇最佳動作，不保存狀態
            self.dqn.epsilon = 0.0  # 在推理模式下關閉探索
            action = self.dqn.act(state)
        
        # 將動作轉換為方向
        if action == 0:  # 保持當前方向
            return intersection.allowed_direction if intersection.allowed_direction else "Horizontal"
        elif action == 1:  # 垂直方向
            return "Vertical"
        else:  # 水平方向
            return "Horizontal"
    
    def get_reward(self, intersection, prev_state, action, current_state, tick):
        """
        計算獎勵值
        
        Args:
            intersection: 交叉路口對象
            prev_state: 之前的狀態
            action: 執行的動作
            current_state: 當前狀態
            tick: 當前時間刻
            
        Returns:
            float: 獎勵值
        """
        # 1. 等待時間變化 (舊的平均等待時間 - 新的平均等待時間)
        prev_h_wait = prev_state[6] * 50.0  # 反歸一化
        prev_v_wait = prev_state[7] * 50.0
        curr_h_wait = current_state[6] * 50.0
        curr_v_wait = current_state[7] * 50.0
        
        wait_time_change = ((prev_h_wait + prev_v_wait) - (curr_h_wait + curr_v_wait)) / 2.0
        
        # 2. 信號燈切換懲罰 (鼓勵穩定性)
        prev_dir = int(prev_state[0] * 2)  # 之前的方向
        curr_dir = int(current_state[0] * 2)  # 當前的方向
        
        switch_penalty = -2.0 if prev_dir != curr_dir else 0.0
        
        # 3. 能源消耗懲罰
        # 計算當前交叉口所有機器人的能源消耗總和
        energy_consumption = 0
        for robot in list(intersection.horizontal_robots.values()) + list(intersection.vertical_robots.values()):
            if hasattr(robot, 'current_intersection_energy_consumption'):
                energy_consumption += robot.current_intersection_energy_consumption
        
        energy_penalty = -0.1 * energy_consumption
        
        # 4. 停止-啟動懲罰
        # 計算當前交叉口所有機器人的停止-啟動次數總和
        stop_and_go_count = 0
        for robot in list(intersection.horizontal_robots.values()) + list(intersection.vertical_robots.values()):
            if hasattr(robot, 'current_intersection_stop_and_go'):
                stop_and_go_count += robot.current_intersection_stop_and_go
        
        stop_go_penalty = -0.5 * stop_and_go_count
        
        # 5. 機器人通過獎勵
        robots_passed = len(intersection.previous_horizontal_robots) + len(intersection.previous_vertical_robots)
        passing_reward = 1.0 * robots_passed
        
        # 總獎勵
        reward = wait_time_change + switch_penalty + energy_penalty + stop_go_penalty + passing_reward
        
        return reward
    
    def train(self, intersection, tick, warehouse):
        """
        訓練DQN模型
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
        """
        intersection_id = intersection.id
        
        # 只有在有先前狀態時才進行訓練
        if intersection_id not in self.previous_states or intersection_id not in self.previous_actions:
            return
        
        prev_state = self.previous_states[intersection_id]
        prev_action = self.previous_actions[intersection_id]
        
        # 獲取當前狀態
        current_state = self.get_state(intersection, tick, warehouse)
        
        # 計算獎勵
        reward = self.get_reward(intersection, prev_state, prev_action, current_state, tick)
        
        # 判斷是否為結束狀態 (如果沒有機器人，或是定期重置)
        done = (len(intersection.horizontal_robots) == 0 and len(intersection.vertical_robots) == 0) or (tick % 1000 == 0)
        
        # 存儲經驗
        self.dqn.remember(prev_state, prev_action, reward, current_state, done)
        
        # 清除先前狀態
        if done:
            del self.previous_states[intersection_id]
            del self.previous_actions[intersection_id]
        else:
            # 更新先前狀態
            self.previous_states[intersection_id] = current_state
        
        # 每64個tick進行一次批次訓練
        if tick % 64 == 0:
            self.dqn.replay(batch_size=32)
        
        # 每1000個tick更新目標網絡
        if tick % 1000 == 0:
            self.dqn.update_target_model()
            
        # 每5000個tick保存模型
        if tick % 5000 == 0 and tick > 0:
            self.dqn.save_model(tick=tick)
    
    def set_training_mode(self, is_training):
        """
        設置是否處於訓練模式
        
        Args:
            is_training (bool): 是否處於訓練模式
        """
        self.is_training = is_training
        if not is_training:
            self.dqn.epsilon = 0.0  # 關閉探索
        
    def load_model(self, model_path=None, tick=None):
        """
        加載預訓練模型
        
        Args:
            model_path (str, optional): 模型路徑
            tick (int, optional): 特定時間點的模型
            
        Returns:
            bool: 是否成功加載模型
        """
        if tick is not None:
            model_path = f"models/{self.model_name}_{tick}.pth"
        
        return self.dqn.load_model(model_path)
