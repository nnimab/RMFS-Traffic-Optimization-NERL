# RMFS 機器人移動履行系統 - 交通控制優化

本項目應用神經進化強化學習（Neuroevolution Reinforcement Learning, NERL）優化機器人移動履行系統（Robot Movement Fulfillment System, RMFS）的交通管理，特別聚焦於降低機器人的能源消耗並維持高效的任務處理。

## 研究背景

RMFS系統是現代智能倉儲的核心組件，其效率直接影響物流運營成本。在大型倉庫中，機器人交通管理尤為重要，良好的交通控制策略可以：
- 減少機器人能源消耗
- 降低停止-啟動頻率
- 避免交通堵塞
- 提高訂單處理效率

## 主要功能

### 交通控制策略
項目實現了多種交通控制策略進行對比實驗：

1. **基於時間的控制器**
   - 採用固定時間週期控制交叉路口方向
   - 考慮水平和垂直方向的不同流量特性
   - 參數可調：水平和垂直方向的綠燈時間

2. **基於隊列的控制器**
   - 根據交叉路口各方向的機器人數量動態調整交通流向
   - 考慮方向偏好因子，反映倉庫佈局特性
   - 設置最小綠燈時間，避免頻繁切換
   - 參數可調：最小綠燈時間和方向偏好因子

3. **DQN控制器**
   - 基於深度Q學習的交通控制
   - 透過強化學習自適應最佳交通策略
   - 參數可調：探索率

4. **神經進化強化學習(NERL)控制器** (開發中)
   - 結合神經網絡與進化算法
   - 通過族群演化尋找最優控制策略
   - 以能源效率為主要優化目標

### 模擬環境

- 基於NetLogo的可視化界面
- Python實現的核心邏輯和控制策略
- 真實物理模型計算能源消耗
- 全面的訂單生成和處理機制

### 評估框架

- 統一的指標收集和分析系統
- 多場景測試（標準、高密度、變化負載）
- 能源消耗、訂單處理效率和交通流量綜合評估

## 系統架構

```
RMFS/
├── ai/                       # 人工智能和控制器
│   ├── controllers/          # 不同控制器實現
│   │   ├── time_based_controller.py
│   │   ├── queue_based_controller.py
│   │   └── dqn_controller.py
│   └── traffic_controller.py # 控制器基類
├── world/                    # 倉庫和環境模型
│   ├── entities/             # 實體類（機器人、工作站等）
│   ├── managers/             # 管理器類（交叉口、機器人等）
│   └── warehouse.py          # 整體倉庫類
├── evaluation/               # 評估框架
│   ├── metrics_collector.py  # 數據收集
│   └── result_analyzer.py    # 結果分析
├── data/                     # 輸入輸出數據
├── rmfs.nlogo                # NetLogo界面
└── netlogo.py                # NetLogo與Python交互
```

## 使用方法

1. 安裝依賴：`pip install -r requirements.txt`
2. 啟動NetLogo並打開`rmfs.nlogo`
3. 點擊"Setup"初始化環境
4. 選擇所需的交通控制器和參數
5. 點擊"Go"或"Go-forever"運行模擬

## 研究目標

- 比較不同交通控制策略的效能
- 驗證NERL方法在交通管理中的優勢
- 找出能源效率與訂單處理效率的最佳平衡點
- 為智能倉儲系統提供低能耗的交通管理解決方案

## 已知問題與修復

### 2024-03-02 v0.1.3：修復：完善Landscape類的邊界檢查

#### 問題描述
在`landscape.py`中的多個方法缺少完整的邊界檢查，可能導致在特定情況下出現`IndexError: list index out of range`錯誤，特別是在使用控制器的場景下。

#### 錯誤原因
以下方法直接訪問地圖數組但沒有足夠的邊界檢查：
- `getNeighborObject`：完全沒有邊界檢查
- `setObject`：沒有檢查新位置和舊位置是否有效
- `_setObjectNew`：沒有檢查坐標是否在有效範圍內

#### 解決方案
1. 為`getNeighborObject`方法添加了完整的邊界檢查：
   ```python
   def getNeighborObject(self, x, y):
       x_rounded = round(x)
       y_rounded = round(y)
       if x_rounded < 0 or y_rounded < 0 or x_rounded > self.dimension or y_rounded > self.dimension:
           return None
       # ...其餘代碼
   ```

2. 在`setObject`方法中添加了對新位置和舊位置的檢查
3. 在`_setObjectNew`方法中添加了坐標有效性檢查
4. 所有方法現在都會在訪問`self.map`之前確保坐標在有效範圍內（>= 0且 <= dimension）

#### 預防類似問題的建議
1. **統一邊界檢查**：考慮為關鍵數據結構添加包裝方法，統一處理邊界檢查邏輯
2. **防禦性編程**：假設輸入可能無效，並提前檢查和處理
3. **完整測試**：添加單元測試以驗證邊界情況下的行為

### 2024-03-02 v0.1.2：修復：地圖邊界檢查問題

#### 問題描述
在運行模擬時出現以下錯誤：
```
IndexError: list index out of range
```
這個錯誤發生在 `landscape.py` 的 `getNeighborObjectWithRadius` 方法中，當機器人嘗試訪問地圖邊界之外的區域時。

#### 錯誤原因
代碼中缺少足夠的邊界檢查：
1. 在 `getNeighborObjectWithRadius` 方法中，只檢查了坐標的下限（>= 0），但沒有檢查上限
2. 當機器人接近地圖邊緣時，可能計算出超出地圖範圍的坐標
3. 這導致 `self.map[p[0]][p[1]]` 嘗試訪問不存在的數組位置，引發索引越界錯誤

#### 解決方案
1. 在 `landscape.py` 的 `getNeighborObjectWithRadius` 方法中增加對坐標上限的檢查：
   ```python
   if i >= 0 and j >= 0 and i < self.dimension+1 and j < self.dimension+1:
   ```

2. 在 `robot.py` 的 `getNearestRobotConflictCandidate` 方法中增加額外的安全檢查：
   ```python
   pos_x = round(self.pos_x)
   pos_y = round(self.pos_y)
   warehouse_dimension = self.robot_manager.warehouse.DIMENSION
   if pos_x < 0 or pos_x >= warehouse_dimension or pos_y < 0 or pos_y >= warehouse_dimension:
       return None
   ```

#### 預防類似問題的建議
1. **邊界檢查**：所有涉及數組索引的操作都應該包含完整的邊界檢查
2. **防禦性編程**：假設輸入可能無效，並提前檢查和處理
3. **單元測試**：為邊界情況編寫測試，特別是涉及地圖邊緣的機器人行為
4. **日誌記錄**：在關鍵點添加日誌，以便在問題發生時更容易診斷

### 2024-03-02 v0.1.1：修復：Station ID 屬性命名不一致問題

#### 問題描述
在運行模擬時出現以下錯誤：
```
AttributeError: 'Picker' object has no attribute 'station_id'
```
這個錯誤發生在 `station_manager.py` 的 `findHighestSimilarityStation` 方法中，當系統嘗試使用 `station.station_id` 屬性而不是正確的 `station.id` 屬性時。

#### 錯誤原因
代碼中存在命名不一致的問題：
1. `Object` 基類（Station 的父類）定義了 `id` 屬性（格式為 `f"{object_type}-{id}"`，如 "picker-0"）
2. 在 `station_manager.py` 中錯誤地使用了 `station.station_id` 而不是 `station.id`
3. 並非所有實體都有 `station_id` 屬性，它只存在於 `Order` 和 `Job` 類中

#### 解決方案
1. 將 `station_manager.py` 中的 `station.station_id` 全部改為 `station.id`
2. 保持 `Order` 和 `Job` 類中使用的 `station_id` 不變，因為它們是正確的

#### 預防類似問題的建議
1. **統一命名規範**：所有實體類屬性應遵循一致的命名模式
2. **加強代碼審查**：特別關注不同對象間的屬性引用
3. **型別提示**：使用 Python 的型別提示功能，有助於在開發階段發現類似問題
4. **單元測試**：增加測試覆蓋率，特別是對象之間的交互測試
