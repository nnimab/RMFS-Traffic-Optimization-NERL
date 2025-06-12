from ai.traffic_controller import TrafficController
import torch
import torch.nn as nn
import numpy as np
import random
import os
import copy
from multiprocessing import Pool
import time

# 使用與DQN相同的QNetwork架構，但用於進化算法而非梯度下降
class EvolvableNetwork(nn.Module):
    """可進化的神經網絡模型"""
    
    def __init__(self, state_size, action_size):
        """
        初始化可進化網絡
        
        Args:
            state_size (int): 狀態空間維度
            action_size (int): 動作空間維度
        """
        super(EvolvableNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x (Tensor): 輸入狀態張量
            
        Returns:
            Tensor: 策略輸出張量
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_weights_as_vector(self):
        """
        將神經網絡權重提取為一維向量
        
        Returns:
            numpy.ndarray: 包含所有權重的一維數組
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.view(-1).cpu().numpy())
        return np.concatenate(weights)
    
    def set_weights_from_vector(self, weight_vector):
        """
        從一維向量設置神經網絡權重
        
        Args:
            weight_vector (numpy.ndarray): 包含所有權重的一維數組
        """
        start = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                weight_vector[start:start+param_size],
                dtype=param.dtype
            ).view_as(param.data)
            start += param_size


class NEController(TrafficController):
    """
    基於神經進化的交通控制器
    
    使用進化算法而非梯度下降來訓練神經網絡策略
    """
    
    def __init__(self, min_green_time=1, bias_factor=1.5, state_size=8, action_size=3, 
                 max_wait_threshold=50, model_name="nerl_traffic", 
                 population_size=40, elite_size=8, tournament_size=4,
                 crossover_rate=0.7, mutation_rate=0.15, mutation_strength=0.1,
                 evolution_interval=100, **kwargs):
        """
        初始化NERL控制器
        
        Args:
            min_green_time (int): 最小綠燈持續時間
            bias_factor (float): 方向偏好因子
            state_size (int): 狀態空間維度
            action_size (int): 動作空間維度
            max_wait_threshold (int): 最大等待閾值
            model_name (str): 模型名稱
            population_size (int): 族群大小
            elite_size (int): 精英保留數量
            tournament_size (int): 錦標賽選擇大小
            crossover_rate (float): 交叉概率
            mutation_rate (float): 變異概率
            mutation_strength (float): 變異強度
            evolution_interval (int): 進化間隔步數
        """
        super().__init__(controller_name="NERL控制器")
        self.min_green_time = min_green_time
        self.bias_factor = bias_factor
        self.state_size = state_size
        self.action_size = action_size
        self.max_wait_threshold = max_wait_threshold
        self.model_name = model_name
        
        # 進化參數
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.evolution_interval = evolution_interval
        
        # 定義任務優先級權重 (與DQN控制器相同)
        self.priority_weights = {
            "delivering_pod": 3.0,  # 送pod去撿貨站 (最高優先級)
            "returning_pod": 2.0,   # 將pod送回倉庫 (次高優先級)
            "taking_pod": 1.0,      # 空車機器人去倉庫拿pod (一般優先級)
            "idle": 0.5,            # 閒置狀態 (最低優先級)
            "station_processing": 0.0  # 在站台處理中的機器人不需要考慮
        }
        
        # 初始化族群
        self.population = self._initialize_population()
        self.fitness_scores = [0] * population_size
        self.current_individual_idx = 0  # 當前正在評估的個體索引
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        # 新增：用於追蹤適應度歷史
        self.best_fitness_history = []
        self.average_fitness_history = []
        
        # 用於存儲每個交叉路口的先前狀態和動作
        self.previous_states = {}
        self.previous_actions = {}
        
        # 評估計數器
        self.steps_since_evolution = 0
        self.evaluation_episodes = {}  # 用於跟踪每個個體的評估數據
        
        # 新增的參數，用於追蹤評估狀態
        self.consecutive_no_evaluation = 0  # 連續沒有評估的次數
        self.individual_eval_time = 0  # 當前個體的評估時間
        
        # 是否處於訓練模式
        self.is_training = True
        
        # 嘗試加載最佳模型
        self.load_model()
    
    def _initialize_population(self):
        """
        初始化族群
        
        Returns:
            list: 包含population_size個神經網絡的列表
        """
        population = []
        for _ in range(self.population_size):
            network = EvolvableNetwork(self.state_size, self.action_size)
            # 使用Xavier初始化權重
            for param in network.parameters():
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            population.append(network)
        return population
    
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
        # 使用與DQN控制器相同的狀態表示
        
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
        
        # 使用當前評估的神經網絡選擇動作
        state = self.get_state(intersection, tick, warehouse)
        
        # 保存當前狀態用於後續學習
        intersection_id = intersection.id
        self.previous_states[intersection_id] = state
        
        # 選擇當前使用的神經網絡
        if self.is_training:
            network = self.population[self.current_individual_idx]
        else:
            network = self.best_individual if self.best_individual is not None else self.population[0]
        
        # 使用網絡預測動作
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = network(state_tensor)
            action = torch.argmax(q_values).item()
        
        self.previous_actions[intersection_id] = action
        
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
        # 使用與DQN相同的獎勵計算方式
        
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
        訓練NERL模型 - 收集獎勵並在適當時機進行進化
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
        """
        if not self.is_training:
            return
        
        intersection_id = intersection.id
        
        # 增加個體評估時間計數
        self.individual_eval_time += 1
        
        # 如果當前個體評估時間過長（超過500 ticks）且沒有得到任何評估，嘗試切換到下一個個體
        if self.individual_eval_time > 500 and self.current_individual_idx not in self.evaluation_episodes:
            self.current_individual_idx = (self.current_individual_idx + 1) % self.population_size
            print(f"Individual evaluation timeout. Switching to individual {self.current_individual_idx}")
            self.individual_eval_time = 0
        
        # 只有在有前一個狀態和動作的情況下才能計算獎勵
        if intersection_id in self.previous_states and intersection_id in self.previous_actions:
            prev_state = self.previous_states[intersection_id]
            action = self.previous_actions[intersection_id]
            current_state = self.get_state(intersection, tick, warehouse)
            
            # 計算獎勵
            reward = self.get_reward(intersection, prev_state, action, current_state, tick)
            
            # 累積當前個體的獎勵
            individual_id = self.current_individual_idx
            if individual_id not in self.evaluation_episodes:
                self.evaluation_episodes[individual_id] = {
                    'total_reward': 0.0,
                    'episode_length': 0,
                    'intersections': {}
                }
            
            if intersection_id not in self.evaluation_episodes[individual_id]['intersections']:
                self.evaluation_episodes[individual_id]['intersections'][intersection_id] = 0.0
            
            self.evaluation_episodes[individual_id]['total_reward'] += reward
            self.evaluation_episodes[individual_id]['intersections'][intersection_id] += reward
            self.evaluation_episodes[individual_id]['episode_length'] += 1
            
            # 更新前一個狀態和動作
            self.previous_states[intersection_id] = current_state
        else:
            # 新增診斷日誌 <- 註解掉
            # if tick % 100 == 0 and intersection_id not in self.previous_states:
            #     print(f"Debug: Intersection {intersection_id} has no previous state at tick {tick}")
            # if tick % 100 == 0 and intersection_id not in self.previous_actions:
            #     print(f"Debug: Intersection {intersection_id} has no previous action at tick {tick}")
            pass # 添加 pass 以維持縮排結構
    
    def _evolve(self):
        """
        執行一代進化
        
        1. 評估當前族群
        2. 選擇最佳個體
        3. 執行交叉和變異產生新一代
        """
        # 將累積的評估結果轉換為適應度分數
        for individual_id, data in self.evaluation_episodes.items():
            if data['episode_length'] > 0:
                # 計算平均獎勵作為適應度分數
                self.fitness_scores[individual_id] = data['total_reward'] / data['episode_length']
            
        # 如果有至少一個個體被評估
        if len(self.evaluation_episodes) > 0:
            # 找出最佳個體
            if any(self.fitness_scores):
                current_best_idx = np.argmax(self.fitness_scores)
                current_best_fitness = self.fitness_scores[current_best_idx]
                
                # 更新全局最佳個體 (如果有更好的)
                if current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_individual = copy.deepcopy(self.population[current_best_idx])
                    print(f"New best individual found with fitness: {self.best_fitness}")
                    
                    # 保存最佳模型
                    self.save_model()
            
            # 新增：在重置前計算平均適應度
            valid_fitness_scores = [score for score in self.fitness_scores if score != 0] # 排除未評估個體的預設值0
            average_fitness = np.mean(valid_fitness_scores) if valid_fitness_scores else 0

            # 創建新一代
            new_population = self._create_new_generation()
            self.population = new_population
            
            # 重置評估數據
            self.fitness_scores = [0] * self.population_size
            self.evaluation_episodes = {}
            self.current_individual_idx = 0
            self.individual_eval_time = 0  # 重置個體評估時間
            self.consecutive_no_evaluation = 0  # 重置連續無評估計數
            
            # 新增：記錄適應度歷史
            self.best_fitness_history.append(self.best_fitness)
            self.average_fitness_history.append(average_fitness)
            
            print(f"Evolution completed. Best fitness: {self.best_fitness}, Average fitness: {average_fitness}")
        else:
            print("No individuals evaluated yet, skipping evolution")
            # 如果多次進化都沒有評估，嘗試更換當前評估的個體
            self.consecutive_no_evaluation += 1
            if self.consecutive_no_evaluation >= 3:
                self.current_individual_idx = (self.current_individual_idx + 1) % self.population_size
                self.individual_eval_time = 0  # 重置新個體的評估時間
                self.consecutive_no_evaluation = 0
                print(f"Consecutive evolution failures. Switching to individual: {self.current_individual_idx}")
            else:
                self.current_individual_idx = (self.current_individual_idx + 1) % self.population_size
                self.individual_eval_time = 0  # 重置新個體的評估時間
                print(f"Switching to next individual: {self.current_individual_idx}")
    
    def _tournament_selection(self, k=3):
        """
        錦標賽選擇
        
        Args:
            k (int): 錦標賽大小
            
        Returns:
            int: 選中的個體索引
        """
        # 隨機選擇k個個體
        selected_indices = np.random.choice(self.population_size, size=k, replace=False)
        # 返回適應度最高的
        return selected_indices[np.argmax([self.fitness_scores[i] for i in selected_indices])]
    
    def _crossover(self, parent1, parent2):
        """
        均勻交叉兩個父代個體
        
        Args:
            parent1: 第一個父代個體
            parent2: 第二個父代個體
            
        Returns:
            EvolvableNetwork: 子代個體
        """
        child = EvolvableNetwork(self.state_size, self.action_size)
        
        # 獲取父代權重
        p1_weights = parent1.get_weights_as_vector()
        p2_weights = parent2.get_weights_as_vector()
        
        # 均勻交叉
        mask = np.random.random(p1_weights.shape) < 0.5
        child_weights = np.where(mask, p1_weights, p2_weights)
        
        # 設置子代權重
        child.set_weights_from_vector(child_weights)
        
        return child
    
    def _mutate(self, individual):
        """
        高斯變異
        
        Args:
            individual: 要變異的個體
            
        Returns:
            EvolvableNetwork: 變異後的個體
        """
        # 獲取權重
        weights = individual.get_weights_as_vector()
        
        # 生成變異遮罩 (選擇哪些權重進行變異)
        mutation_mask = np.random.random(weights.shape) < self.mutation_rate
        
        # 生成高斯噪聲
        noise = np.random.normal(0, self.mutation_strength, weights.shape)
        
        # 應用變異
        weights = weights + mutation_mask * noise
        
        # 設置變異後的權重
        individual.set_weights_from_vector(weights)
        
        return individual
    
    def _create_new_generation(self):
        """
        創建新一代族群
        
        Returns:
            list: 新族群
        """
        new_population = []
        
        # 精英保留 - 直接複製最佳個體
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(self.population[idx]))
        
        # 填充剩餘族群
        while len(new_population) < self.population_size:
            # 決定是否進行交叉
            if random.random() < self.crossover_rate and len(new_population) + 1 < self.population_size:
                # 交叉
                parent1_idx = self._tournament_selection(self.tournament_size)
                parent2_idx = self._tournament_selection(self.tournament_size)
                
                child1 = self._crossover(self.population[parent1_idx], self.population[parent2_idx])
                child2 = self._crossover(self.population[parent2_idx], self.population[parent1_idx])
                
                # 變異
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                # 只複製和變異
                parent_idx = self._tournament_selection(self.tournament_size)
                child = copy.deepcopy(self.population[parent_idx])
                child = self._mutate(child)
                new_population.append(child)
        
        return new_population
    
    def set_training_mode(self, is_training):
        """
        設置是否處於訓練模式
        
        Args:
            is_training (bool): 是否訓練
        """
        self.is_training = is_training
        print(f"NERL controller set to {'training' if is_training else 'inference'} mode")
    
    def load_model(self, model_path=None, tick=None):
        """
        加載已保存的模型
        
        Args:
            model_path (str, optional): 完整模型路徑，如果未提供則使用默認模型
            tick (int, optional): 時間刻，用於加載特定時間點的模型
            
        Returns:
            bool: 是否成功加載模型
        """
        if not model_path:
            model_path = f"models/{self.model_name}.pth"
            if tick is not None:
                model_path = f"models/{self.model_name}_{tick}.pth"
                
        if os.path.isfile(model_path):
            try:
                # 加載最佳個體
                network = EvolvableNetwork(self.state_size, self.action_size)
                network.load_state_dict(torch.load(model_path))
                self.best_individual = network
                print(f"NERL model loaded: {model_path}")
                return True
            except Exception as e:
                print(f"Error loading NERL model: {e}")
                return False
        else:
            print(f"NERL model file not found: {model_path}")
            return False
    
    def save_model(self, model_name=None, tick=None):
        """
        保存最佳模型
        
        Args:
            model_name (str, optional): 模型名稱，默認為None使用self.model_name
            tick (int, optional): 當前時間節點，用於標識保存的模型版本
        """
        if self.best_individual is None:
            print("No best individual to save yet")
            return
            
        # 創建models目錄（如果不存在）
        os.makedirs("models", exist_ok=True)
        
        # 確定保存路徑
        save_name = model_name if model_name else self.model_name
        if tick is not None:
            save_name = f"{save_name}_{tick}"
        
        save_path = f"models/{save_name}.pth"
        
        try:
            torch.save(self.best_individual.state_dict(), save_path)
            print(f"NERL model saved: {save_path}")
        except Exception as e:
            print(f"Error saving NERL model: {e}")

    def step_evolution_counter_and_evolve(self, tick):
        """
        增加進化計數器，並在達到間隔時執行進化。
        此方法應在每個模擬 tick 結束時（處理完所有交叉口後）調用一次。
        
        Args:
            tick: 當前時間刻
        """
        if not self.is_training:
            return
            
        self.steps_since_evolution += 1
        
        if self.steps_since_evolution >= self.evolution_interval:
            if tick % 100 == 0: # 保留調試日誌
                print(f"Debug: Evolution attempt at tick {tick} (interval: {self.evolution_interval}), evaluated individuals: {len(self.evaluation_episodes)}")
            self._evolve()
            self.steps_since_evolution = 0 # 重置計數器 