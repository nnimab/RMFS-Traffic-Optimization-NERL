# RMFS 機器人移動履行系統 - 交通控制優化

本項目應用神經進化強化學習（Neuroevolution Reinforcement Learning, NERL）優化機器人移動履行系統（Robot Movement Fulfillment System, RMFS）的交通管理，特別聚焦於降低機器人的能源消耗並維持高效的任務處理。

## 研究背景

RMFS系統是現代智能倉儲的核心組件，其效率直接影響物流運營成本。在大型倉庫中，機器人交通管理尤為重要，良好的交通控制策略可以：

-   減少機器人能源消耗
-   降低停止-啟動頻率
-   避免交通堵塞
-   提高訂單處理效率

## 研究目標

-   比較不同交通控制策略的效能
-   驗證NERL方法在交通管理中的優勢
-   找出能源效率與訂單處理效率的最佳平衡點
-   為智能倉儲系統提供低能耗的交通管理解決方案

## 主要功能

### 交通控制策略

項目實現了多種交通控制策略進行對比實驗：

**架構說明：** 所有的交通控制策略（Time-based, Queue-based, DQN, NERL）都採用一種 **邏輯上分散、實例上集中** 的架構。系統中存在一個中央的 `IntersectionManager`，它管理著所有交叉口。當需要進行交通決策時，`IntersectionManager` 會遍歷每一個交叉口，並調用**當前選定的控制器實例**（例如，一個 `DQNController` 或 `NEController` 實例）的決策方法（如 `get_direction`）。控制器根據**該特定交叉口**的狀態資訊（如排隊車輛、等待時間等）獨立做出決策。

-   **優點：** 方便統一管理和切換不同控制策略，便於集中收集數據和訓練（對於DQN/NERL）。
-   **注意：** 雖然是單一控制器實例，但其內部邏輯是針對每個交叉口獨立運算的，利用每個交叉口的本地信息進行決策。

1.  **基於時間的控制器**

    -   採用固定時間週期控制交叉路口方向
    -   考慮水平和垂直方向的不同流量特性
    -   參數可調：水平和垂直方向的綠燈時間
    -   *運作方式：* 由中央管理器為每個交叉口計算當前時間週期應允許的方向。

2.  **基於隊列的控制器**

    -   根據交叉路口各方向的機器人數量和優先級動態調整交通流向
    -   考慮方向偏好因子，反映倉庫佈局特性
    -   設置最小綠燈時間，避免頻繁切換
    -   參數可調：最小綠燈時間和方向偏好因子
    -   *運作方式：* 由中央管理器為每個交叉口計算其隊列加權優先級，決定方向。
    -   **綠波機制**：
        -   協調相鄰交叉口的交通控制，使機器人能連續通過多個路口。
        -   參數：綠波影響因子 (green\_wave\_influence, 建議值為 5.0)，綠波傳播時間 (green\_wave\_propagation\_time, 建議值為 15)。
        -   **短距離交叉路口優化**：
            -   使用平方根距離計算綠波衰減。
            -   增加基礎影響值，確保短距離路口也能有效協調。
            -   增加交叉路口相鄰識別距離到 15 個單位。
    -   **單向交通優化**：
        -   當只有一個方向有機器人時，立即允許該方向通行。
    -   **機器人狀態權重**:
        -    優先考慮運送中任務的機器人。

3.  **DQN控制器**

    -   基於深度Q學習的交通控制器，結合規則邏輯與強化學習能力
    -   參數可調：
        -   `min_green_time`: 最小綠燈持續時間，避免頻繁切換（默認1tick）
        -   `bias_factor`: 方向偏好因子，調整水平和垂直方向的權重（默認1.5）
        -   `max_wait_threshold`: 最大等待時間閾值，用於防鎖死機制（默認50ticks）
        -   `model_name`: 模型名稱，用於保存和加載（默認"dqn_traffic"）
    -   *運作方式：* 中央管理器為每個交叉口調用DQN控制器的 `get_direction`。控制器獲取該交叉口的狀態，使用神經網絡預測最佳動作（保持、切換水平、切換垂直），並在訓練模式下根據獎勵更新網絡。內部狀態（如 `previous_states`）按交叉口ID區分。
    -   **狀態空間設計**：
        -   8維狀態向量，包含以下特徵：
        -   當前允許的方向編碼（None=0, Vertical=1, Horizontal=2）
        -   自上次信號變化以來的時間（ticks）
        -   水平/垂直方向的機器人數量
        -   水平/垂直方向的優先級機器人數量（delivering_pod狀態）
        -   水平/垂直方向的平均等待時間
    -   **動作空間設計**：
        -   3個可能的動作：保持當前信號狀態、切換到水平方向通行、切換到垂直方向通行
    -   **獎勵函數設計**：
        -   等待時間減少（正獎勵）
        -   信號燈切換懲罰（鼓勵穩定性）
        -   能源消耗懲罰
        -   停止-啟動次數懲罰
        -   機器人通過獎勵
    -   **神經網絡架構**：
        -   使用全連接網絡(MLP)
        -   輸入層：8個神經元（對應狀態空間）
        -   兩個隱藏層：各24個神經元，使用ReLU激活函數
        -   輸出層：3個神經元（對應動作空間）
    -   **模型訓練與保存**：
        -   每64個tick進行一次批次訓練（批次大小32）
        -   每1000個tick更新目標網絡
        -   每5000個tick自動保存模型：`models/{model_name}_{tick}.pth`
        -   支持暫停和繼續訓練功能
    -   **模型加載**：
        -   支持兩種方式加載預訓練模型：
        -   指定模型路徑：`controller.load_model(model_path="models/my_model.pth")`
        -   指定特定tick保存的模型：`controller.load_model(tick=5000)`
    -   **運行模式**：
        -   訓練模式：收集經驗並更新網絡
        -   推理模式：僅使用已訓練網絡做出決策，不進行探索
        -   整合防鎖死機制，避免機器人長時間等待
        -   保留最小綠燈時間限制，避免頻繁切換
        -   當一個方向沒有機器人時，優先選擇有機器人的方向

4.  **神經進化強化學習(NERL)控制器**

    -   結合神經網絡與進化算法，通過族群演化尋找最優控制策略
    -   參數可調：
        -   `min_green_time`: 最小綠燈持續時間（默認1tick）
        -   `bias_factor`: 方向偏好因子（默認1.5）
        -   `max_wait_threshold`: 最大等待時間閾值（默認50ticks）
        -   `model_name`: 模型名稱（默認"nerl_traffic"）
        -   `population_size`: 種群規模（默認40）
        -   `elite_size`: 精英個體數量（默認8）
        -   `tournament_size`: 錦標賽選擇規模（默認4）
        -   `crossover_rate`: 交叉率（默認0.7）
        -   `mutation_rate`: 變異率（默認0.15）
        -   `mutation_strength`: 變異強度（默認0.1）
        -   `evolution_interval`: 進化間隔（默認每15個tick）
    -   *運作方式：* 與DQN類似，中央管理器為每個交叉口調用NERL控制器的 `get_direction`。控制器使用**當前種群中的最佳個體**（或在訓練模式下是特定個體）的神經網絡，根據該交叉口的狀態做出決策。進化過程作用於整個種群，評估基於所有交叉口的總體性能。
    -   **狀態空間設計**：
        -   與DQN控制器相同，8維狀態向量
    -   **動作空間設計**：
        -   與DQN控制器相同，3個可能動作
    -   **獎勵函數設計**：
        -   等待時間減少（正獎勵）
        -   信號切換懲罰
        -   能源消耗懲罰
        -   機器人通過獎勵
    -   **神經網絡架構**：
        -   進化網絡（EvolvableNetwork）
        -   輸入層：8個神經元（對應狀態空間）
        -   隱藏層：24個神經元，ReLU激活
        -   輸出層：3個神經元（對應動作空間）
    -   **進化機制**：
        -   精英保留：每代保留8個最佳個體
        -   錦標賽選擇：隨機選擇4個個體，選出最優的作為父代
        -   均勻交叉：兩個父代基因均勻交叉產生子代
        -   高斯變異：隨機添加高斯噪聲到權重
    -   **模型訓練與保存**：
        -   每15個tick評估一次種群
        -   使用篩選和進化更新種群
        -   每5000個tick保存最佳個體：`models/{model_name}_{tick}.pth`
    -   **模型加載**：
        -   支持直接加載已保存的模型：`controller.load_model(tick=5000)`
    -   **運行模式**：
        -   訓練模式：持續進行種群進化
        -   評估模式：僅使用最佳個體做出決策，不進行進化

### 模擬環境

-   基於NetLogo的可視化界面
-   Python實現的核心邏輯和控制策略
-   真實物理模型計算能源消耗
-   全面的訂單生成和處理機制

### 評估框架

-   統一的指標收集和分析系統
-   多場景測試（標準、高密度、變化負載）
-   能源消耗、訂單處理效率和交通流量綜合評估

## 系統架構

```
RMFS/
├── ai/                       # 人工智能和控制器
│   ├── controllers/          # 不同控制器實現
│   │   ├── time_based_controller.py
│   │   ├── queue_based_controller.py
│   │   ├── dqn_controller.py
│   │   └── nerl_controller.py
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

1.  安裝依賴：`pip install -r requirements.txt`
2.  啟動NetLogo並打開`rmfs.nlogo`
3.  點擊"Setup"初始化環境
4.  選擇所需的交通控制器和參數
    -   **DQN 模型加載**：
        -   可通過 `model-tick` 滑桿選擇要加載的模型，點擊 `DQN(加載模型)` 按鈕加載。
        -   點擊 `查看模型` 按鈕查看可用模型。
    -   **NERL 控制器使用**：
        -   **通過界面按鈕**：
            -   點擊 `NERL控制器` 按鈕設置基本NERL控制器
            -   調整 `exploration-rate` 和 `model-tick` 滑桿，點擊 `NERL(加載模型)` 按鈕加載預訓練模型
            -   點擊 `NERL訓練模式開` 切換到訓練模式，點擊 `關` 切換到評估模式
        -   **通過命令行**：
            -   設置NERL控制器：`set-nerl-controller`
            -   加載預訓練模型：`set-nerl-controller exploration-rate 0 load-model-tick 5000`
            -   切換訓練/評估模式：`set-nerl-training-mode false`（評估模式）或 `set-nerl-training-mode true`（訓練模式）
            -   查看可用NERL模型：`list-models-by-type "nerl"`
        -   **評估模式推薦設置**：設置較低的exploration-rate（比如0），加載預訓練模型，然後切換到評估模式
5.  點擊"Go"或"Go-forever"運行模擬
6.  **查看當前 tick**：NetLogo 界面會顯示當前 tick 值。

### 訂單配置參數修改

要修改訂單生成的相關參數，需要編輯 `lib/generator/warehouse_generator.py` 文件中的 `draw_layout_from_generated_file` 函數內的 `config_orders` 調用部分。主要參數說明如下：

#### 普通訂單參數（70-77行）

```python
config_orders(
    initial_order=20,                                      # 初始訂單數量
    total_requested_item=500,                              # 倉庫SKU總數
    items_orders_class_configuration={"A": 0.6, "B": 0.3, "C": 0.1}, # 訂單中各類別物品的比例
    quantity_range=[1, 12],                                # 每個訂單中物品數量範圍
    order_cycle_time=100,                                  # 每小時訂單數量
    order_period_time=5,                                   # 訂單生成總時長（小時）
    order_start_arrival_time=5,                            # 訂單開始到達時間
    date=1,                                                # 日期標記
    sim_ver=1,                                             # 模擬版本標記
    dev_mode=False)                                        # 開發模式開關
```

#### 積壓訂單參數（84-94行）

```python
config_orders(
    initial_order=50,                                      # 積壓訂單初始數量
    total_requested_item=500,                              # 倉庫SKU總數
    items_orders_class_configuration={"A": 0.6, "B": 0.3, "C": 0.1}, # 訂單中各類別物品的比例
    quantity_range=[1, 12],                                # 每個訂單中物品數量範圍
    order_cycle_time=100,                                  # 每小時訂單數量
    order_period_time=3,                                   # 積壓訂單總時長（小時）
    order_start_arrival_time=5,                            # 訂單開始到達時間
    date=1,                                                # 日期標記
    sim_ver=2,                                             # 模擬版本標記（2表示積壓訂單）
    dev_mode=True)                                         # 開發模式開關
```

#### 重要參數說明

-   **initial\_order**: 初始訂單數量，對於普通訂單和積壓訂單分別設置
-   **total\_requested\_item**: 倉庫中可用的SKU總數，通常保持一致
-   **order\_cycle\_time**: 每小時產生的訂單數量，數值越大訂單越密集
-   **order\_period\_time**: 訂單生成的總時間長度（小時），增加此值會生成更多訂單
-   **quantity\_range**: 每個訂單中包含的物品數量範圍，影響訂單複雜度

#### 注意事項

修改參數前，必須刪除以下文件才能使新配置生效：

1.  `data/input/assign_order.csv` - 訂單分配記錄
2.  `data/output/generated_order.csv` - 生成的訂單數據
3.  `data/input/generated_backlog.csv` - 生成的積壓訂單數據
4.  `data/output/items.csv` - 生成的物品數據
5.  `data/output/skus_data.csv` - 生成的SKU數據

如果想完全重置系統，也可刪除：

-   `data/output/generated_pod.csv` - 生成的貨架數據

刪除文件後，重新執行初始化過程（點擊"Setup"按鈕）以生成新的訂單數據。

### 調適訊息控制系統

系統實現了全局的調適訊息控制機制，通過靜態屬性 `Robot.DEBUG_LEVEL` 控制不同級別的調適資訊輸出：

#### 調適級別（DEBUG_LEVEL）說明

-   **級別 0**：不顯示任何調適訊息，適合正式運行環境。
-   **級別 1**：僅顯示重要的警告和訓練資訊，如 DQN 訓練進度、目標網絡更新等。
-   **級別 2**：顯示所有詳細調適資訊，包含機器人運動、交叉路口運作等細節。

#### 使用方法

要修改調適訊息級別，可在代碼中修改 `world/entities/robot.py` 文件的 `DEBUG_LEVEL` 靜態屬性值：

```python
# debug level: 0 = no debug, 1 = important only, 2 = verbose
DEBUG_LEVEL = 1  # 修改為所需級別
```

#### 調適訊息類型

調適系統輸出的主要資訊類型包括：

1.  **訓練資訊**：
    -   DQN 控制器訓練開始
    -   目標網絡更新
    -   模型參數和記憶體狀態

2.  **機器人運動資訊**：
    -   路徑規劃與決策
    -   接近交叉路口的距離判斷
    -   坐標轉換與計算

3.  **系統資訊**：
    -   NetLogo 與 Python 通信事件
    -   系統時鐘更新

所有調適訊息使用英文顯示，以方便程式開發者識別系統運行狀態。

## 已知問題與修復

詳細的問題修復記錄請參見 CHANGELOG.md 文件。

