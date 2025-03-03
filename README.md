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

1.  **基於時間的控制器**

    -   採用固定時間週期控制交叉路口方向
    -   考慮水平和垂直方向的不同流量特性
    -   參數可調：水平和垂直方向的綠燈時間

2.  **基於隊列的控制器**

    -   根據交叉路口各方向的機器人數量動態調整交通流向
    -   考慮方向偏好因子，反映倉庫佈局特性
    -   設置最小綠燈時間，避免頻繁切換
    -   參數可調：最小綠燈時間和方向偏好因子
    -   **綠波機制**：
        -   協調相鄰交叉路口的交通控制，使機器人能連續通過多個路口。
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

    -   基於深度Q學習的交通控制
    -   透過強化學習自適應最佳交通策略
    -   參數可調：探索率
    -   **模型保存與加載**：
        -   自動在特定 ticks (5000, 10000, 20000) 保存模型。
        -   可通過 NetLogo 界面加載指定 tick 的預訓練模型。
    -   **DQN 算法完善**：
        -   實現了完整的 DQN 模型，包括記憶回放、目標網絡、ε-greedy 策略。
        -   狀態空間：8 維向量，包含交叉路口信息。
        -   動作空間：2 維，代表水平和垂直方向。
        -   參數：最小綠燈時間 (10), 折扣因子 (0.95), 探索率 (初始 1.0, 衰減至 0.01), 目標網絡更新頻率 (100 步)。
    -   **訓練訊息**：
        -    訓練過程會顯示相關訊息 (需開啟調試訊息, 見下方說明)。

4.  **神經進化強化學習(NERL)控制器** (開發中)

    -   結合神經網絡與進化算法
    -   通過族群演化尋找最優控制策略
    -   以能源效率為主要優化目標

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

1.  安裝依賴：`pip install -r requirements.txt`
2.  啟動NetLogo並打開`rmfs.nlogo`
3.  點擊"Setup"初始化環境
4.  選擇所需的交通控制器和參數
    -   **DQN 模型加載**：
        -   可通過 `model-tick` 滑桿選擇要加載的模型，點擊 `DQN(加載模型)` 按鈕加載。
        -   點擊 `查看模型` 按鈕查看可用模型。
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

