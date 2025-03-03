# 更新日誌 (Changelog)

本文件包含項目的所有重要更改。

## v0.1.18 (2025-03-02)

### 功能：在 NetLogo 界面顯示當前 tick 值

* **功能描述**：在 NetLogo 界面上添加了顯示當前模擬 tick 值的監視器，方便用戶直觀地了解模擬進度。

* **實現細節**：
  1. 在 `netlogo.py` 的 `tick()` 函數中添加返回 `warehouse._tick` 值：
     - 修改函數返回值列表，增加 tick 值作為結果的一部分
     - 確保 tick 值能夠正確地從 Python 傳遞到 NetLogo 界面

  2. 在 `rmfs.nlogo` 修改：
     - 在 globals 部分添加 `current_tick` 全局變量
     - 修改 `go` 方法，從 Python 結果中獲取 tick 值並設置到全局變量
     - 添加新的 MONITOR 元素顯示 `current_tick` 值

* **使用效果**：
  1. 用戶現在可以在界面上直接看到當前模擬的進度（tick 值）
  2. 有助於與其他模擬指標（如能源消耗、訂單數量等）進行關聯分析
  3. 方便調試和記錄實驗結果時參考具體的時間點

* **修改文件**：
  - `netlogo.py`：修改 `tick()` 函數，返回 tick 值
  - `rmfs.nlogo`：添加全局變量和新的界面顯示元素

## v0.1.17 (2025-02-26)

### 文档：在 README.md 中添加調適訊息控制系統說明

* **功能描述**：在 README.md 中添加了關於 Robot.DEBUG_LEVEL 系統的詳細說明，幫助開發者了解如何控制和使用調適訊息功能。

* **實現細節**：
  1. 在 README.md 中新增了「調適訊息控制系統」章節，包含以下內容：
     - 調適級別（DEBUG_LEVEL）的完整說明（0-無調適、1-重要訊息、2-所有訊息）
     - 如何在代碼中修改 DEBUG_LEVEL 值的實用指南
     - 調適訊息類型的詳細分類（訓練訊息、機器人運動訊息、系統訊息）

  2. 提供了清晰的使用示例，幫助開發者了解如何設置和應用不同的調適級別：
     - 不同場景下的推薦設置（正式環境：0，一般開發：1，深入調試：2）
     - 通過修改 Robot 類中的 DEBUG_LEVEL 靜態屬性來控制全局調適訊息

* **目的與價值**：
  1. 提高開發體驗：開發者可按需控制調適訊息的詳細程度
  2. 標準化調適實踐：建立清晰的調適訊息級別規範
  3. 提升文檔完整性：補充系統調適功能的使用指南
  4. 便於新開發者快速上手：無需閱讀代碼即可了解調適系統

* **修改文件**：
  - `README.md`：添加調適訊息控制系統的詳細說明

## v0.1.16 (2025-02-25)

### 功能：新增 DQN 訓練調適訊息

* **功能描述**：在 DQN 控制器訓練過程中添加調適訊息，方便開發者了解訓練進度和狀態。

* **實現細節**：
  1. 在 `ai/controllers/dqn_controller.py` 中添加訓練開始和目標網絡更新的調適訊息：
     - 在每次訓練開始時輸出訓練步驟和當前時間刻
     - 在目標網絡更新時顯示訊息

  2. 在 `ai/deep_q_network.py` 的 `replay` 方法中添加訓練詳細資訊：
     - 顯示批次大小和記憶庫大小
     - 方便監控訓練樣本的累積情況

* **使用方法**：
  1. 訓練訊息的顯示受 `Robot.DEBUG_LEVEL` 控制，與其他調適訊息一致
  2. 當 `DEBUG_LEVEL > 0` 時顯示訓練相關訊息
  3. 訓練訊息使用英文顯示，以 "[Training]" 作為前綴便於識別

* **修改文件**：
  - `ai/controllers/dqn_controller.py`：添加訓練開始和目標網絡更新訊息
  - `ai/deep_q_network.py`：在 replay 方法中添加訓練詳情訊息

## v0.1.15 (2025-02-24)

### 優化：改進 Robot 調適文字輸出系統

* **問題描述**：Robot 類中的調適訊息（如距離計算、交叉路口檢查等）過多且不加過濾地顯示，干擾了開發和測試過程中的輸出觀察。

* **解決方案**：
  1. 在 `Robot` 類中添加了 `DEBUG_LEVEL` 靜態屬性，用於控制調適訊息的輸出級別：
     - 0：完全不顯示調適訊息
     - 1：只顯示重要調適訊息（如機器人在交叉路口等待時間過長的警告）
     - 2：顯示所有詳細調適訊息

  2. 修改了 `pathBlockedByIntersection` 方法中的調適訊息輸出，根據 `DEBUG_LEVEL` 控制顯示級別。

  3. 修改了 `closeEnough` 方法中的距離檢查調適訊息，只在 `DEBUG_LEVEL > 1` 時顯示。

  4. 修改了 `ensureCoordinate` 方法中的調適訊息，只在 `DEBUG_LEVEL > 1` 時顯示。

  5. 修改了 `netlogo.py` 文件中的 "before tick" 訊息輸出，根據 `Robot.DEBUG_LEVEL` 控制顯示。

* **使用方法**：
  1. 設置 `Robot.DEBUG_LEVEL = 0` 可完全關閉調適訊息，適合正式運行環境。
  2. 設置 `Robot.DEBUG_LEVEL = 1` 可只顯示重要的警告訊息，適合一般開發和測試。
  3. 設置 `Robot.DEBUG_LEVEL = 2` 可顯示所有詳細調適訊息，適合深入調試。

* **預設值**：預設將 `DEBUG_LEVEL` 設為 0，完全關閉調適訊息以獲得清晰的輸出。

* **修改文件**：
  - `world/entities/robot.py`：添加 `DEBUG_LEVEL` 靜態屬性、修改多處 print 語句添加條件判斷
  - `netlogo.py`：修改 `tick` 函數中 "before tick" 訊息的輸出條件

## v0.1.14 (2025-02-23)

### 修復：DQN 控制器缺少 last_state 和 last_action 屬性

* **問題描述**：DQN 控制器在執行過程中出現 `AttributeError: 'DQNController' object has no attribute 'last_state'` 錯誤，導致模擬在 tick 10 左右停止。

* **錯誤原因**：
  1. `DQNController` 類中在 `get_direction` 方法中使用了 `self.last_state` 和 `self.last_action` 屬性。
  2. 但在類的 `__init__` 方法中未初始化這些屬性，導致首次執行時出現 AttributeError。
  3. 這些屬性用於存儲每個交叉路口的先前狀態和動作，用於 DQN 算法中的經驗回放。

* **解決方案**：
  1. 修改 `ai/controllers/dqn_controller.py` 文件中的 `__init__` 方法：
     - 在初始化其他屬性的同時，添加 `self.last_state = {}` 和 `self.last_action = {}` 初始化語句
     - 確保這些屬性被正確初始化為空字典，用於後續存儲交叉路口狀態信息

* **技術細節**：
  - 這兩個屬性是字典類型，使用交叉路口 ID 作為鍵
  - 初始化為空字典，隨著控制器運行逐漸填充各交叉路口的狀態和動作記錄
  - 這些記錄對於實現 DQN 算法中的經驗回放至關重要

* **修改文件**：
  - `ai/controllers/dqn_controller.py`：在 `__init__` 方法中添加 `self.last_state` 和 `self.last_action` 屬性初始化

### 修復：Robot 對象缺少 last_move_tick 屬性問題

* **問題描述**：DQN 控制器在執行過程中出現 `AttributeError: 'Robot' object has no attribute 'last_move_tick'` 錯誤，導致模擬無法繼續運行。

* **錯誤原因**：
  1. 在 `dqn_controller.py` 的 `get_state` 方法中，嘗試訪問 `robot.last_move_tick` 屬性來計算機器人等待時間。
  2. `Robot` 類中並未定義此屬性，導致運行時錯誤。
  3. 這個屬性的目的是為了計算機器人在交叉路口等待的時間，用於 DQN 的狀態向量。

* **解決方案**：
  1. 修改 `ai/controllers/dqn_controller.py` 文件中的 `get_state` 方法：
     - 用 `robot.current_intersection_start_time` 替代不存在的 `last_move_tick` 屬性
     - 添加 null 檢查，確保即使屬性為空也能提供默認值 0
     - 使用列表手動收集等待時間，然後找出最大值，避免列表推導式的潛在問題

* **技術細節**：
  - `current_intersection_start_time` 記錄了機器人進入交叉路口的時間點
  - 通過計算當前時間與進入時間的差值，得到等待時間
  - 對空值進行防禦性處理，確保代碼穩健性
  - 保持狀態向量的維度不變，確保與模型兼容

* **修改文件**：
  - `ai/controllers/dqn_controller.py`：修改 `get_state` 方法中計算等待時間的邏輯

## v0.1.13 (2025-02-22)

### 修復：DQN 控制器中的屬性錯誤問題

* **問題描述**：DQN 控制器在執行過程中出現 `AttributeError: 'Intersection' object has no attribute 'current_intersection_energy_consumption'` 錯誤，導致模擬無法繼續運行。

* **錯誤原因**：
  1. 在 `dqn_controller.py` 的 `get_state` 方法中，嘗試訪問 `intersection.current_intersection_energy_consumption` 和 `intersection.current_intersection_stop_and_go` 屬性。
  2. `Intersection` 類中並未定義這些屬性，導致運行時錯誤。
  3. 這些屬性的目的是取得交叉路口能源消耗和停止啟動次數，但實際實現與設計不一致。

* **解決方案**：
  1. 修改 `ai/controllers/dqn_controller.py` 文件中的 `get_state` 方法：
     - 使用預設值 0.0 替代缺失的能源消耗和停止啟動屬性
     - 確保狀態向量維度一致，保持 DQN 模型的穩定性

  2. 修改 `get_reward` 方法：
     - 替換原本直接訪問 `current_intersection_energy_consumption` 屬性的代碼
     - 改用交叉路口的 `calculateAverageStopAndGo` 方法取得停止啟動統計數據
     - 計算水平和垂直方向的停止啟動次數總和作為獎勵計算依據

* **技術細節**：
  - 狀態向量結構仍維持 8 維，確保與之前訓練的模型兼容
  - 能源消耗暫時設為預設值 0.0，後續可考慮添加能源消耗計算邏輯
  - 停止啟動次數改為基於交叉路口已有的統計函數計算

* **修改文件**：
  - `ai/controllers/dqn_controller.py`：修改 `get_state` 和 `get_reward` 方法

## v0.1.12 (2025-02-21)

### 功能：增強 DQN 控制器的自動保存和加載模型功能

* **功能描述**：為 DQN 控制器添加自動保存里程碑模型功能並增強模型加載機制，方便實驗比較和延續訓練。

* **實現細節**：
  1. 修改了 `ai/controllers/dqn_controller.py` 文件：
     - 添加了在特定 ticks (5000, 10000, 20000) 自動保存模型的功能
     - 增強了 `save_model` 和 `load_model` 方法，支持通過 tick 指定模型版本
     - 優化了模型保存命名規則，清晰區分常規保存和里程碑保存

  2. 修改了 `ai/deep_q_network.py` 文件：
     - 增強了 `load_model` 方法，支持加載指定路徑的模型
     - 添加了模型加載成功與失敗的回傳值，便於上層邏輯處理
     - 優化了模型存儲結構和錯誤處理機制

  3. 修改了 `netlogo.py` 文件：
     - 增強了 `set_dqn_controller` 函數，支持在初始化控制器時加載指定 tick 的模型
     - 添加了 `list_available_models` 函數，用於查詢可用的已保存模型
     - 完善了錯誤處理和日誌輸出，便於診斷模型加載問題

  4. 修改了 `rmfs.nlogo` 文件：
     - 添加了 `model-tick` 滑桿，用於選擇要加載的模型 tick
     - 添加了 `查看模型` 按鈕，用於顯示所有可用的模型
     - 添加了 `DQN(加載模型)` 按鈕，用於加載指定 tick 的預訓練模型
     - 添加了相應的 NetLogo 過程處理模型加載和列表功能

* **使用方法**：
  1. 正常訓練過程中，系統將自動在 5000, 10000, 20000 ticks 保存里程碑模型
  2. 要查看可用的模型，點擊 `查看模型` 按鈕
  3. 要加載預訓練模型，滑動 `model-tick` 滑桿選擇要加載的 tick 值，然後點擊 `DQN(加載模型)` 按鈕
  4. 加載成功後，控制器顯示為 "DQN(loaded)"，表示使用的是已訓練的模型

* **技術注意事項**：
  1. 里程碑模型使用 `dqn_traffic_milestone_{tick}.pth` 命名格式存儲
  2. 每 1000 步的常規保存仍然保留，使用 `dqn_traffic_{tick}.pth` 命名格式
  3. 所有模型都存儲在 `models` 目錄下，確保此目錄的存在和可寫性
  4. 加載模型時會同時更新主網絡和目標網絡，確保一致性

* **修改文件**：
  - `ai/controllers/dqn_controller.py`：增強了模型保存和加載功能
  - `ai/deep_q_network.py`：完善了模型加載機制和錯誤處理
  - `netlogo.py`：添加了模型加載和列表功能
  - `rmfs.nlogo`：更新了界面，添加了模型選擇和加載控件

## v0.1.11 (2025-02-20)

### 優化：完善 DQN 控制器實現

* **功能描述**：完善 DQN (Deep Q-Network) 控制器實現，從簡單的隨機控制策略升級為完整的深度強化學習控制器。

* **實現細節**：
  1. 擴展了 `ai/deep_q_network.py` 文件，實現了完整的 DQN 模型：
     - 增加了 `QNetwork` 神經網絡類，定義了三層神經網絡架構
     - 完善了 `DeepQNetwork` 類，實現了記憶回放、目標網絡、ε-greedy 策略等核心 DQN 功能
     - 增加了模型保存和加載功能，支持訓練中斷和恢復
     - 實現了批量訓練和經驗回放機制，提高學習效率

  2. 重構了 `ai/controllers/dqn_controller.py` 中的 `DQNController` 類：
     - 實現了複雜的狀態特徵提取，包括交叉路口方向、等待機器人數量、能源消耗等
     - 設計了獎勵函數，結合通過機器人數量、能源消耗和停止啟動次數
     - 集成了定期訓練和模型更新機制
     - 添加了自適應探索機制，平衡探索與利用

* **參數配置**：
  1. 狀態空間（state_size）：8 維向量，包括當前方向、持續時間、水平/垂直機器人數量等
  2. 動作空間（action_size）：2，代表 "Horizontal" 和 "Vertical" 兩個方向
  3. 最小綠燈時間（min_green_time）：10 個時間單位，避免頻繁切換方向
  4. 折扣因子（gamma）：0.95，用於平衡當前和未來獎勵
  5. 探索率（epsilon）：初始 1.0，隨時間衰減至 0.01，平衡探索與利用
  6. 目標網絡更新頻率（update_target_every）：100 步

* **預期效果**：
  1. 控制器能夠學習最優的交通控制策略，隨時間提高性能
  2. 系統能自適應地應對不同的交通模式和負載
  3. 平衡效率、能源消耗和停止啟動次數等多個目標
  4. 通過經驗積累不斷改進，逐步超越簡單的啟發式方法

* **技術注意事項**：
  1. 為提高訓練穩定性，使用了目標網絡和主網絡分離的策略
  2. 狀態特徵進行了規範化處理，確保神經網絡輸入穩定
  3. 實現了模型定期保存機制，每 1000 步保存一次
  4. 完整支持 PyTorch 和 CPU/GPU 訓練

* **修改文件**：
  - `ai/deep_q_network.py`：完善了神經網絡模型和 DQN 算法實現
  - `ai/controllers/dqn_controller.py`：完善了 DQN 控制器實現和特徵提取

## v0.1.10 (2025-02-19)

### 優化：進一步增強綠波機制效果解決短距離交叉路口問題

* **問題描述**：優化後的綠波機制（v0.1.9）運行效果仍不理想，尤其是當交叉路口之間距離較短時，機器人仍然在通過一個路口後在下一個路口停下來，降低了運輸效率。

* **問題根因分析**：
  1. 綠波傳播速度問題：綠波「到達」下一個路口的時間與機器人到達的時間不夠同步。
  2. 相鄰路口識別問題：對於距離很短的路口，系統未能正確將它們識別為關鍵的相鄰關係。
  3. 距離衰減過快：原有公式中距離因子的影響過大，導致即使是短距離的相鄰路口間綠波效應也衰減嚴重。
  4. 路口間綠波協調不足：對於特別近的路口，需要更強的綠波影響力。

* **解決方案**：
  1. 將綠波影響因子（green_wave_influence）從 3.0 進一步增加到 5.0，大幅加強綠波對交通方向決策的影響力。
  2. 將綠波傳播時間（green_wave_propagation_time）從 10 個時間單位增加到 15 個時間單位，確保綠波效應能持續更長時間。
  3. 修改距離衰減計算方式，使用平方根距離而非線性距離，降低距離對綠波影響的抑制作用。
  4. 增加基礎影響值（0.5），確保即使距離較遠的路口也能接收到一定程度的綠波影響。
  5. 增加交叉路口相鄰識別距離，從 10 個單位增加到 15 個單位，以便更好地識別和處理短距離路口。
  6. 加強調試輸出，增加詳細的綠波傳播記錄，便於診斷和優化。

* **技術實現細節**：
  - 修改了 `ai/controllers/queue_based_controller.py` 中的綠波影響計算公式:
    ```python
    # 原來: influence = self.green_wave_influence * (1.0 - time_since_change / self.green_wave_propagation_time) / max(1, distance)
    # 現在:
    time_factor = 1.0 - time_since_change / self.green_wave_propagation_time
    distance_factor = 1.0 / max(1, (distance ** 0.5))  # 使用平方根降低距離影響
    base_influence = 0.5  # 基礎影響值
    influence = self.green_wave_influence * time_factor * (distance_factor + base_influence)
    ```
  - 擴大了相鄰交叉路口的識別半徑（從10到15），確保短距離路口能被正確關聯
  - 增加了詳細的綠波傳播記錄，包括影響值、距離、時間等信息

* **預期效果**：
  1. 機器人能夠更流暢地連續通過多個交叉路口，特別是距離較短的相鄰路口
  2. 綠波效應在相鄰路口間傳播更加高效，協調性更強
  3. 整體運輸效率得到進一步提升，減少在短距離路口的停止和等待時間
  4. 系統能適應不同密度的交叉路口分佈，包括非常接近的交叉路口

## v0.1.9 (2025-02-18)

### 優化：加強綠波機制效果與交通控制參數

* **問題描述**：即使實現了綠波機制，機器人仍然無法連續通過多個交叉路口，在經過一個路口後在下一個路口就停下來，降低了運輸效率。

* **優化原因**：
  1. 綠波影響因子設置過低，綠波協調效果不明顯。
  2. 綠波傳播時間太短，無法有效覆蓋機器人從一個交叉路口移動到下一個交叉路口所需的時間。
  3. 最小綠燈時間不足，導致交通信號燈切換過於頻繁，機器人沒有足夠時間通過交叉路口。

* **解決方案**：
  1. 將綠波影響因子（green_wave_influence）從 1.8 增加到 3.0，加強綠波對交通方向決策的影響力。
  2. 將綠波傳播時間（green_wave_propagation_time）從 4 個時間單位增加到 10 個時間單位，確保綠波效應能持續足夠長時間。
  3. 將最小綠燈時間（min_green_time）從 10 增加到 30，給予機器人足夠的通行時間窗口。
  4. 保留水平方向偏好因子（bias_factor），繼續為水平方向提供 1.5 倍的權重優勢。

* **預期效果**：
  1. 機器人能夠連續通過多個交叉路口，減少不必要的停止和等待。
  2. 交通信號燈狀態更加穩定，減少頻繁切換。
  3. 綠波效應能夠有效傳播到相鄰的多個交叉路口。
  4. 整體運輸效率得到顯著提升。

* **技術細節**：
  - 這些參數優化保持了原有交通控制邏輯的完整性，僅調整了關鍵數值參數。
  - 參數調整基於實際運行觀察結果，針對特定的交通流模式進行了優化。
  - 通過增加綠波影響和持續時間，系統能夠更好地協調相鄰交叉路口的交通控制。

## v0.1.8 (2025-02-17)

### 修復：解決 GridPosition 對象序列化問題

* **問題描述**：系統在嘗試保存狀態時出現 `_pickle.PicklingError: Can't pickle <class 'ai.controllers.queue_based_controller.GridPosition'>: attribute lookup GridPosition on ai.controllers.queue_based_controller failed` 錯誤，導致無法完成模擬執行。

* **錯誤原因**：
  1. 在 `initialize_adjacent_intersections` 方法內部使用 `namedtuple` 動態創建了 `GridPosition` 類。
  2. Python 的 `pickle` 模組在序列化時無法在模組級別找到 `GridPosition` 的定義。
  3. 在多個地方重複定義了相同的 `GridPosition` 類，導致類定義不一致。
  4. `namedtuple` 類型在序列化時可能出現問題，特別是當它在動態環境中使用時。

* **解決方案**：
  1. 用自定義的 `GridPosition` 類替換 `namedtuple`，添加完整的類方法和序列化支持。
  2. 實現 `__reduce__` 方法，確保正確的序列化和反序列化。
  3. 添加 `__eq__`、`__hash__` 和 `__repr__` 方法，保持與 `namedtuple` 相同的功能。
  4. 移除方法內部重複的 `GridPosition` 定義，統一使用模組級別的定義。

* **技術細節**：
  - 在模組級別定義了完整的 `GridPosition` 類，取代原來的 `namedtuple`。
  - 實現了 `__reduce__` 方法來支持 `pickle` 序列化，返回 `(GridPosition, (self.x, self.y))`。
  - 添加了比較方法和哈希計算，確保類的實例可以正確地用於字典鍵和集合。
  - 保持與原有代碼的兼容性，確保現有的 `grid_position` 賦值和訪問方式不受影響。

## v0.1.7 (2025-02-16)

### 修復：綠波機制適配性問題

* **問題描述**：綠波機制實現後系統出現 `AttributeError: 'list' object has no attribute 'values'` 錯誤，以及交叉路口缺少 `grid_position` 屬性的警告，導致系統無法正確建立相鄰關係。

* **錯誤原因**：
  1. 在 `initialize_adjacent_intersections` 方法中假設了 `intersection_manager.intersections` 必須是字典類型，而在實際環境中它可能是列表。
  2. 缺少異常處理機制，導致系統在遇到錯誤時直接崩潰。
  3. 多處代碼使用了可能不存在的屬性或方法，沒有進行充分的防禦性編程。
  4. 某些交叉路口沒有 `grid_position` 屬性，但代碼直接訪問此屬性而沒有檢查其存在性。

* **解決方案**：
  1. 修改 `initialize_adjacent_intersections` 方法，增加對 `intersections` 為列表或字典的兼容處理。
  2. 添加全面的異常處理機制，確保在出現問題時系統能夠優雅降級。
  3. 增加調試輸出，幫助診斷綠波機制的工作狀態。
  4. 改進 `track_robot_movement` 和 `get_direction` 方法，增強錯誤處理能力。
  5. 添加屬性存在性檢查，避免訪問不存在的屬性。
  6. 為缺少 `grid_position` 屬性的交叉路口動態創建此屬性，基於其已有的 `coordinate` 或 `pos_x/pos_y` 屬性。

* **技術細節**：
  - 使用 `isinstance()` 函數判斷數據類型，根據不同類型採取不同的處理策略
  - 對於可能失敗的操作使用 `try-except` 塊進行保護
  - 如果初始化失敗，自動禁用綠波功能以避免影響基本交通控制
  - 使用 `getattr()` 安全訪問對象屬性，避免直接屬性訪問可能引發的錯誤
  - 添加回退機制，確保在出現異常時仍能返回合理的默認值
  - 使用 `namedtuple` 動態創建 `grid_position` 屬性，保持與原有屬性同步
  - 增強統計和日誌輸出，記錄缺少屬性的交叉路口數量和建立的連接數

## v0.1.6 (2025-02-15)

### 功能：實現交通綠波協調機制

* **功能描述**：實現綠波（Green Wave）機制，協調多個相鄰交叉路口的交通控制，使機器人能夠連續通過多個交叉路口而不需頻繁停止等待。

* **實現細節**：
  1. 修改 `ai/controllers/queue_based_controller.py` 中的 `QueueBasedController` 類，添加綠波協調功能。
  2. 建立交叉路口相鄰關係映射，識別水平和垂直方向的相鄰路口。
  3. 新增機器人移動軌跡跟踪，記錄機器人經過的交叉路口和方向。
  4. 開發綠波影響因子計算方法，基於相鄰路口狀態、距離和時間關係。
  5. 在交通方向決策邏輯中整合綠波影響權重，增強相鄰路口間的方向協調。

* **參數配置**：
  1. 綠波影響因子（green_wave_influence）：1.8，用於平衡綠波協調與基本交通邏輯。
  2. 綠波傳播時間（green_wave_propagation_time）：4個時間單位，定義綠波效應的持續時間。
  3. 可通過構造函數參數 `green_wave_enabled` 控制是否啟用綠波功能。

* **預期效果**：
  1. 減少機器人走走停停的情況，提高整體運輸效率。
  2. 對於沿同一方向行駛的多個機器人，形成"綠色通道"。
  3. 在保持現有交通控制優勢的同時，引入智能化的路口協調機制。
  4. 系統能自適應地識別主要交通流方向，優先保障主幹道的通行效率。

* **技術注意事項**：
  1. 綠波機制不會覆蓋最小綠燈時間保護，確保交通規則的基本穩定性。
  2. 相鄰路口定義基於坐標距離，最大考慮10個單位距離內的交叉路口。
  3. 加入調試輸出，記錄綠波影響值，便於後續優化。

## v0.1.5 (2025-02-14)

### 修復：強化機器人交叉路口判斷和通行邏輯

* **問題描述**：即使重新設計了隊列基控制器，機器人在某些情況下仍然會卡在交叉路口不動。

* **錯誤原因**：
  1. 在 `pathBlockedByIntersection` 方法中，機器人判斷是否可以通過路口的邏輯過於簡單。
  2. 沒有處理長時間等待的情況，機器人可能無限期地卡在路口。
  3. 距離判斷邏輯（`closeEnough`）可能不夠精確。

* **解決方案**：
  1. 修改 `world/entities/robot.py` 中的 `pathBlockedByIntersection` 方法，添加更詳細的邏輯和調試信息輸出。
  2. 新增等待時間計數機制：每個機器人現在會記錄在每個路口等待的時間。
  3. 添加安全機制：當機器人在路口等待超過30個時間單位時，強制允許通行。
  4. 使用更嚴格的距離判斷標準（從1.0減少到0.8）。

* **技術細節**：
  - 在 `Robot` 類中添加了 `intersection_wait_time` 字典屬性，記錄機器人在每個路口等待的時間
  - 改進了 `pathBlockedByIntersection` 方法，添加了詳細的調試信息和等待時間處理
  - 當機器人等待時間過長時，會強制允許通行，以防止永久卡住的情況
  - 添加了記錄和重置等待時間的邏輯

## v0.1.4 (2025-02-13)

### 改進：重新設計隊列基控制器

* **問題描述**：使用 queue_based_controller.py 時，機器人在交叉路口不動，尤其是當只有一個方向有機器人時。

* **錯誤原因**：
  1. 原始控制器沒有特別處理「只有一個方向有機器人」的情況。
  2. 沒有考慮機器人的任務狀態優先級（如運送中、取貨中等）。
  3. 只計算等待機器人數量，沒有考慮任務重要性。

* **解決方案**：
  1. 在 `ai/controllers/queue_based_controller.py` 中重寫控制器邏輯。
  2. 添加特殊邏輯：當只有一個方向有機器人時，立即允許該方向通行。
  3. 引入機器人狀態權重系統，優先考慮運送任務的機器人。
  4. 改進權重計算方法，用於決定哪個方向應該被允許通行。
  5. 當權重相同且當前沒有設定方向時，預設選擇水平方向。

* **技術細節**：
  - 添加了 `status_weights` 字典，為不同的機器人狀態分配權重
  - 修改 `get_direction` 方法，先檢查是否只有單向交通
  - 使用加權計算替代簡單的計數比較

## v0.1.3 (2025-02-12)

### 修復：完善Landscape類的邊界檢查

* **問題描述**：在`landscape.py`中的多個方法缺少完整的邊界檢查，可能導致在特定情況下出現`IndexError: list index out of range`錯誤。

* **錯誤原因**：以下方法直接訪問地圖數組但沒有足夠的邊界檢查：
  - `getNeighborObject`：完全沒有邊界檢查
  - `setObject`：沒有檢查新位置和舊位置是否有效
  - `_setObjectNew`：沒有檢查坐標是否在有效範圍內

* **解決方案**：
  - 為`getNeighborObject`方法添加了完整的邊界檢查
  - 在`setObject`方法中添加了對新位置和舊位置的檢查
  - 在`_setObjectNew`方法中添加了坐標有效性檢查
  - 所有方法現在都會在訪問`self.map`之前確保坐標在有效範圍內（>= 0且 <= dimension）

* **預防類似問題的建議**：
  - 所有直接訪問數組的操作應該包含完整的邊界檢查
  - 考慮為關鍵數據結構添加包裝方法，統一處理邊界檢查邏輯
  - 添加單元測試以驗證邊界情況下的行為

## v0.1.2 (2025-02-11)

### 修復：地圖邊界檢查問題

* **問題描述**：在運行模擬時出現以下錯誤：
  ```
  IndexError: list index out of range
  ```
  這個錯誤發生在`landscape.py`的`getNeighborObjectWithRadius`方法中，當機器人嘗試訪問地圖邊界之外的區域時。

* **錯誤原因**：
  1. 在`getNeighborObjectWithRadius`方法中，只檢查了坐標的下限（>= 0），但沒有檢查上限
  2. 當機器人接近地圖邊緣時，可能計算出超出地圖範圍的坐標
  3. 這導致`self.map[p[0]][p[1]]`嘗試訪問不存在的數組位置，引發索引越界錯誤

* **解決方案**：
  1. 在`landscape.py`的`getNeighborObjectWithRadius`方法中增加對坐標上限的檢查：
     ```python
     if i >= 0 and j >= 0 and i < self.dimension+1 and j < self.dimension+1:
     ```
  2. 在`robot.py`的`getNearestRobotConflictCandidate`方法中增加額外的安全檢查。

* **預防類似問題的建議**：
  1. 邊界檢查：所有涉及數組索引的操作都應該包含完整的邊界檢查
  2. 防禦性編程：假設輸入可能無效，並提前檢查和處理
  3. 單元測試：為邊界情況編寫測試，特別是涉及地圖邊緣的機器人行為
  4. 日誌記錄：在關鍵點添加日誌，以便在問題發生時更容易診斷

## v0.1.1 (2025-02-10)

### 修復：Station ID 屬性命名不一致問題

* **問題描述**：在運行模擬時出現以下錯誤：
  ```
  AttributeError: 'Picker' object has no attribute 'station_id'
  ```
  這個錯誤發生在`station_manager.py`的`findHighestSimilarityStation`方法中，當系統嘗試使用`station.station_id`屬性而不是正確的`station.id`屬性時。

* **錯誤原因**：
  1. `Object`基類（Station的父類）定義了`id`屬性（格式為`f"{object_type}-{id}"`，如"picker-0"）
  2. 在`station_manager.py`中錯誤地使用了`station.station_id`而不是`station.id`
  3. 並非所有實體都有`station_id`屬性，它只存在於`Order`和`Job`類中

* **解決方案**：
  1. 將`station_manager.py`中的`station.station_id`全部改為`station.id`
  2. 保持`Order`和`Job`類中使用的`station_id`不變，因為它們是正確的

* **預防類似問題的建議**：
  1. 統一命名規範：所有實體類屬性應遵循一致的命名模式
  2. 加強代碼審查：特別關注不同對象間的屬性引用
  3. 型別提示：使用Python的型別提示功能，有助於在開發階段發現類似問題
  4. 單元測試：增加測試覆蓋率，特別是對象之間的交互測試