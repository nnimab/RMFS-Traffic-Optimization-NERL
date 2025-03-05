# NERL應用於RMFS機器人交通優化研究計畫

## 研究背景

### 神經進化強化學習 (NERL) 簡介

神經進化強化學習（Neuroevolution Reinforcement Learning, NERL）結合神經網路與進化演算法，用演化的方法訓練強化學習代理的策略。不同於傳統深度強化學習通過梯度下降調整神經網路權重，NERL以演化方式搜尋最優策略。

NERL的特點：
- 在族群中同時評估多個策略（個體）
- 根據累積回饋（適應度）選出較佳者進行交叉與變異
- 不依賴梯度資訊，對獎勵延遲或稀疏的問題特別有利
- 避免傳統策略梯度法容易陷入的局部極值
- 易於平行化，在大規模問題上訓練速度更快

### 研究目標

應用NERL優化機器人移動履行系統(RMFS)的交通管理，特別聚焦於：
- 維持相同的任務處理效率
- 降低機器人的能源消耗
- 優化交叉路口的交通流量

## 當前系統分析

### 代碼結構

當前RMFS系統基於NetLogo和Python實現：
- NetLogo提供視覺化界面和基礎模擬環境
- Python處理核心邏輯和控制策略
- 主要組件包括：
  - `Warehouse`類：系統核心控制器
  - `RobotManager`、`IntersectionManager`等管理器類
  - `Robot`、`Intersection`、`Pod`等實體類

### 能源消耗模型

系統中的能源消耗計算基於以下物理因素：
```python
def calculateEnergy(self, velocity, acceleration):
    tick_unit = TICK_TO_SECOND
    if acceleration != 0 and velocity != 0:
        average_speed = 2 * velocity + (acceleration * tick_unit)
        return (self.MASS + self.LOAD_MASS) * ((self.GRAVITY * self.FRICTION) + (
                acceleration * self.INERTIA)) * average_speed * tick_unit / 7200
    elif velocity != 0:
        return (self.MASS + self.LOAD_MASS) * self.GRAVITY * self.FRICTION * velocity * tick_unit / 3600
    return 0
```

關鍵參數：
- 機器人質量 (MASS)
- 負載質量 (LOAD_MASS)
- 重力 (GRAVITY)
- 摩擦力 (FRICTION)
- 慣性 (INERTIA)

系統還追踪：
- 停止-啟動次數 (`stop_and_go`)
- 轉彎次數 (`total_turning`)

### 交通管理框架

當前系統包含一個基本的強化學習框架，但尚未完全實現：
- `IntersectionManager`類負責交叉路口管理
- 存在`updateDirectionUsingDQN`方法
- `deep_q_network.py`提供基本架構但核心功能未實現

## 三種對照組設計

為了評估NERL的效果，我們將實現三種交通控制策略作為對照組：

### 1. 基於時間的交通控制

特點：
- 簡單的時間週期控制
- 水平方向（左右）因pods數量較多，設置更長的綠燈時間
- 不考慮實時交通狀況

```python
class TimeBasedController(TrafficController):
    def __init__(self, horizontal_green_time=70, vertical_green_time=30):
        self.horizontal_green_time = horizontal_green_time  # 水平方向綠燈時間更長
        self.vertical_green_time = vertical_green_time
        self.cycle_length = horizontal_green_time + vertical_green_time
    
    def get_direction(self, intersection, tick, warehouse):
        # 計算當前週期內的時間位置
        cycle_position = tick % self.cycle_length
        
        # 水平方向有更長的綠燈時間
        if cycle_position < self.horizontal_green_time:
            return "horizontal"
        else:
            return "vertical"
```

### 2. 基於隊列長度的交通控制（啟發式方法）

特點：
- 根據等待機器人數量調整交通方向
- 為水平方向設置偏好因子，反映pods分佈特性
- 設置最小綠燈時間，避免頻繁切換

```python
class QueueBasedController(TrafficController):
    def __init__(self, min_green_time=10, bias_factor=1.5):
        self.min_green_time = min_green_time  # 最小綠燈時間
        self.bias_factor = bias_factor  # 水平方向偏好因子
        self.last_change = {}  # 記錄每個交叉路口上次更改的時間
        self.current_direction = {}  # 當前方向
    
    def get_direction(self, intersection, tick, warehouse):
        # 獲取水平和垂直方向的隊列長度
        horizontal_count = len(intersection.horizontal_robots)
        vertical_count = len(intersection.vertical_robots)
        
        # 考慮方向偏好 - 水平方向有優勢
        horizontal_count = horizontal_count * self.bias_factor
        
        # 根據隊列長度決定方向
        # ...
```

### 3. 傳統深度Q學習 (DQN)

特點：
- 基於強化學習，自適應學習最優策略
- 考慮複雜的狀態特徵
- 能隨時間優化策略

```python
class DQNController(TrafficController):
    def __init__(self, state_size=10, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        # ...
    
    def get_state(self, intersection, tick, warehouse):
        # 獲取交叉口狀態
        state = [
            1 if intersection.allowed_direction == "horizontal" else 0,
            intersection.durationSinceLastChange(tick),
            len(intersection.horizontal_robots),
            # ...
        ]
        return np.array(state)
    
    def get_reward(self, intersection, warehouse):
        # 計算獎勵函數
        energy_since_last = intersection.current_intersection_energy_consumption
        stops_since_last = intersection.current_intersection_stop_and_go
        
        # 獎勵 = -(能源消耗 + 停止啟動懲罰)
        reward = -(energy_since_last + 0.5 * stops_since_last)
        
        return reward


後續建議
參數調優：
控制器的 min_green_time、bias_factor 和 max_wait_threshold 參數
DQN的學習率、折扣因子和探索率
獎勵函數的各項權重
模型評估：
與基於隊列的控制器進行對比測試
評估不同交通條件下的性能
進階功能：
實現優先權調整
增加更多特徵如任務緊急程度
考慮相鄰交叉路口之間的協調


        
```

## NERL實現框架

在上述三種對照組的基礎上，我們將實現NERL方法：

### 神經網絡設計

```python
class NEController(TrafficController):
    def __init__(self, population_size=50, state_size=10, action_size=2):
        self.population_size = population_size
        self.state_size = state_size
        self.action_size = action_size
        self.population = self._initialize_population()
        self.best_individual = None
        
    def _initialize_population(self):
        # 初始化神經網絡族群
        population = []
        for _ in range(self.population_size):
            nn = NeuralNetwork(
                input_size=self.state_size,
                hidden_layers=[24, 24],
                output_size=self.action_size
            )
            population.append(nn)
        return population
```

### 適應度函數設計

```python
def fitness_function(warehouse, control_strategy, simulation_time=1000):
    # 重置模擬環境
    warehouse.reset()
    
    # 應用控制策略並運行模擬
    for _ in range(simulation_time):
        # 對每個交叉路口應用策略
        for intersection in warehouse.intersection_manager.intersections:
            state = warehouse.intersection_manager.getState(intersection, warehouse._tick)
            action = control_strategy.forward(torch.FloatTensor(state))
            direction = intersection.getAllowedDirectionByCode(torch.argmax(action).item())
            warehouse.intersection_manager.updateAllowedDirection(intersection.id, direction, warehouse._tick)
        
        # 模擬一個時間步
        warehouse.tick()
    
    # 計算適應度分數
    # 能源效率 = 完成的任務數 / 總能源消耗
    energy_efficiency = len([j for j in warehouse.job_manager.jobs if j.is_finished]) / (warehouse.total_energy + 1e-10)
    
    # 考慮停止-啟動和轉彎的懲罰
    stop_go_penalty = 0.1 * warehouse.stop_and_go
    turning_penalty = 0.05 * warehouse.total_turning
    
    # 最終適應度分數
    fitness = energy_efficiency - stop_go_penalty - turning_penalty
    
    return fitness
```

### 進化操作設計

```python
def evolve(population, warehouse, generations=50):
    for generation in range(generations):
        # 評估適應度
        fitness_scores = []
        for individual in population:
            fitness = fitness_function(warehouse, individual)
            fitness_scores.append((individual, fitness))
        
        # 排序
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 打印當前代的最佳適應度
        print(f"Generation {generation}: Best Fitness = {fitness_scores[0][1]}")
        
        # 選擇前50%作為父代
        parents = [ind for ind, _ in fitness_scores[:len(population)//2]]
        
        # 生成下一代
        next_generation = parents.copy()  # 精英保留
        
        # 通過交叉和變異生成剩餘個體
        while len(next_generation) < len(population):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=0.1)
            next_generation.append(child)
        
        population = next_generation
    
    # 返回最佳個體
    return population[0]
```

## 實驗設計

### 統一評估框架

為確保公平比較，我們將使用統一的評估框架，包括：

1. **相同的輸入數據結構**：
   - 所有控制器接收完全相同格式的交叉路口信息
   - 相同的狀態表示方式（機器人數量、等待時間等）

2. **相同的輸出要求**：
   - 所有控制器都必須輸出相同格式的決策
   - 使用相同的方法應用這些決策

3. **相同的評估環境**：
   - 相同的倉庫配置
   - 相同的訂單生成模式
   - 相同的模擬時長
   - 相同的初始條件

4. **相同的指標收集**：
   - 對所有方法收集完全相同的性能指標
   - 使用相同的頻率收集數據
   - 使用相同的計算方法計算衍生指標

### 評估指標

1. **效率指標**：
   - 完成訂單數量
   - 平均訂單完成時間
   - 吞吐量（單位時間完成訂單數）

2. **能源指標**：
   - 總能耗
   - 平均每訂單能耗
   - 能源效率（完成訂單數/總能耗）

3. **交通指標**：
   - 停止-啟動次數
   - 轉彎次數
   - 平均等待時間

### 實驗場景

1. **標準工作負載**：默認訂單生成率
2. **高密度工作負載**：訂單生成率提高50%
3. **變化工作負載**：訂單生成率週期性變化

## 下一步行動計劃

1. **實現基礎框架**：
   - 創建`TrafficController`基類和工廠方法
   - 修改`IntersectionManager`支持控制器切換

2. **實現三種對照組**：
   - 基於時間的控制器
   - 基於隊列的控制器
   - DQN控制器

3. **建立基準測試**：
   - 實現數據收集和分析工具
   - 運行基準測試並記錄結果

4. **實現NERL方法**：
   - 定義神經網絡結構
   - 實現進化算法操作
   - 設計適應度函數

5. **執行比較實驗**：
   - 在不同場景下測試各方法
   - 收集和分析結果
   - 調整和優化NERL參數

6. **結果可視化和分析**：
   - 創建比較圖表
   - 分析能源效率改進
   - 總結最佳策略和參數設置 