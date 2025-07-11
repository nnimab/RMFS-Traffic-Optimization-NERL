# **RMFS 機器人移動履行系統 - 交通控制優化 (修訂版)**

本項目旨在應用**神經進化強化學習（Neuroevolution Reinforcement Learning, NERL）** 等先進方法，優化**機器人移動履行系統（Robot Movement Fulfillment System, RMFS）** 中的交通管理。核心目標是顯著降低倉庫內自主移動機器人（AMR）的**能源消耗**，同時維持甚至提升整體的**任務處理效率**。

## 研究背景

RMFS 已成為現代智能倉儲與物流中心不可或缺的核心組件。其運作效率直接關係到物流成本、訂單響應速度及客戶滿意度。在規模龐大、機器人數量眾多的倉庫環境中，高效的交通管理變得至關重要。一個精心設計的交通控制策略能夠帶來多重效益：

* **降低能源消耗**：減少不必要的加速、減速和怠速等待，優化行駛路徑。
* **減少停止-啟動頻率**：降低機器人電機和傳動系統的磨損，並進一步節省能源。
* **避免交通堵塞**：預防或快速疏解交叉口的擁堵，保障物流暢通。
* **提高訂單處理效率**：縮短機器人完成任務（如取貨、送貨）的總時間，提升倉庫吞吐量。

## 研究目標

本項目設定以下具體研究目標：

1.  **效能比較**：設計、實現並客觀比較多種不同交通控制策略（基於規則、基於學習）在模擬環境中的表現。
2.  **NERL 驗證**：重點評估和驗證 NERL 方法在 RMFS 交通管理任務上的潛力和優勢，特別是在能源效率和處理效率方面的表現。
3.  **尋找平衡點**：探索不同策略下，**能源效率**與**訂單處理效率**之間的權衡關係。**注意**：本項目旨在提供詳盡的性能指標（KPIs），使使用者能根據自身業務需求，進行後續分析（如設定閾值、加權求和、繪製帕累托前沿）來**自行定義和尋找**最適合其運營目標的「最佳平衡點」。系統本身不直接輸出單一的「平衡分數」。
4.  **提供解決方案**：基於研究結果，為智能倉儲系統提供一套或多套經過驗證的、具備低能耗特性的交通管理策略建議與實現。

## 主要功能

### 交通控制策略

項目實現並對比了以下幾種交通控制策略：

**核心架構說明：** 本系統所有交通控制策略（Time-based, Queue-based, DQN, NERL）均採用一種 **邏輯上分散 (Logically Decentralized)、實例上集中 (Instance-wise Centralized)** 的架構。存在一個中央的 `IntersectionManager` 實例，負責管理倉庫內的所有交叉口。當模擬進入需要進行交通決策的時刻，`IntersectionManager` 會迭代遍歷每一個交叉口。對於每個交叉口，它會調用**當前系統選定的單一控制器實例**（例如，同一個 `DQNController` 或 `NEController` 實例）的決策方法（如 `get_direction`）。該控制器實例會接收**特定交叉口的當前狀態資訊**（如排隊車輛數、等待時間、當前信號狀態等），並**針對該交叉口獨立地**計算出最佳決策（例如，維持當前方向或切換方向）。

* **優點：** 這種架構便於統一管理和切換不同的控制策略，有利於集中收集數據進行分析，並且簡化了基於學習的控制器（DQN/NERL）的訓練過程（因為模型和記憶體是集中的）。
* **關鍵理解：** 儘管物理上只有一個控制器對象，但其內部決策邏輯是分散到每個交叉口的——即每個決策都是基於該交叉口的本地信息獨立做出的。這與完全集中的、需要全局信息的控制器不同，也與每個交叉口擁有完全獨立的模型和記憶體的多智能體系統（MAS）架構有所區別。

---

1.  **基於時間的控制器 (Time-based Controller)**
    * **原理：** 採用預設的固定時間週期，輪流給予交叉口的水平和垂直方向通行權。
    * **特性：** 可考慮水平和垂直方向可能存在的不同主導交通流量，設定不同的綠燈時間。
    * **可調參數：**
        * `horizontal_green_time`: 水平方向的綠燈持續時間 (ticks)。
        * `vertical_green_time`: 垂直方向的綠燈持續時間 (ticks)。
    * **運作方式：** 由中央 `IntersectionManager` 根據全局時間模除週期總長，為每個交叉口確定當前應允許的通行方向。

2.  **基於隊列的控制器 (Queue-based Controller)**
    * **原理：** 根據交叉口各個方向等待的機器人數量、機器人狀態（優先級）以及可選的倉庫佈局偏好，動態地決定哪個方向應該獲得通行權。
    * **特性：**
        * **動態調整：** 相較於固定時間，更能適應實時變化的交通流量。
        * **考慮優先級：** 可以為執行特定任務（如運送貨物中）的機器人賦予更高的通行權重。
        * **方向偏好：** 允許根據倉庫主通道或特定佈局設置方向偏好因子。
        * **最小綠燈時間：** 避免過於頻繁的信號切換，增加交通流的穩定性。
        * **單向交通優化：** 當只有一個方向有機器人等待時，立即將通行權賦予該方向，提高效率。
        * **機器人狀態權重：** 優先考慮處於 `delivering_pod` 狀態的機器人。
    * **可調參數：**
        * `min_green_time`: 最小綠燈持續時間 (ticks)。
        * `bias_factor`: 方向偏好因子，用於調整水平與垂直方向的基礎權重。
    * **運作方式：** 由中央 `IntersectionManager` 為每個交叉口調用此控制器的 `get_direction` 方法。該方法計算各方向的加權隊列長度（考慮數量、優先級、偏好），並結合最小綠燈時間約束來決定是否切換方向。
    * **綠波機制 (Green Wave Mechanism) - 重要澄清：** 經程式碼確認 (A. 控制器細節 - 基於隊列)，先前文檔中描述的基於相鄰路口協調、傳播時間、距離衰減的**綠波機制並未在當前代碼中實現**。此控制器目前僅基於**本地交叉口**的信息（隊列長度、等待時間、優先級、偏好因子、最小綠燈時間）進行決策。

3.  **DQN 控制器 (Deep Q-Network Controller)**
    * **原理：** 採用深度強化學習中的 Q-Learning 算法。控制器學習一個 Q 函數，用於預測在給定交叉口狀態下，執行每個可能動作（保持、切換水平、切換垂直）能獲得的長期累積獎勵期望值，並選擇期望值最高的動作。
    * **特性：** 結合了基於規則的邏輯（如最小綠燈時間、防鎖死機制）和強化學習的自適應決策能力。
    * **可調參數：**
        * `min_green_time`: 最小綠燈持續時間 (預設 1 tick)。
        * `bias_factor`: 方向偏好因子 (預設 1.5)。
        * `max_wait_threshold`: 最大等待時間閾值，觸發防鎖死機制 (預設 50 ticks)。
        * `model_name`: 用於保存和加載訓練模型的名稱 (預設 "dqn_traffic")。
        * (以及 DQN 算法本身的超參數，如學習率、折扣因子、探索率 ε、經驗回放池大小、批次大小、目標網絡更新頻率等，通常在 `deep_q_network.py` 中設定)。
    * **運作方式：** 中央 `IntersectionManager` 為每個交叉口調用 DQN 控制器的 `get_direction` 方法。控制器獲取該交叉口的狀態，將狀態輸入神經網絡得到各動作的 Q 值。在訓練模式下，會根據探索策略（如 ε-greedy）選擇動作；在推理模式下，則直接選擇 Q 值最高的動作。控制器內部按交叉口 ID 區分和管理狀態（如 `previous_states`）。
    * **狀態空間設計 (State Space - 8 維向量):**
        * **構成 (經歸一化處理)：**
            1.  `dir_code / 2.0`: 當前允許方向編碼 (None/H=0, V=0.5, H=1.0 - *注意實際歸一化方式可能與註解略有差異，但目標是區分三個狀態*)。
            2.  `time_since_last_change / 20.0`: 自上次信號變化以來經過的時間 (ticks)，歸一化。
            3.  `horizontal_queue_count / 10.0`: 水平方向等待的機器人數量，歸一化。
            4.  `vertical_queue_count / 10.0`: 垂直方向等待的機器人數量，歸一化。
            5.  `horizontal_priority_ratio`: 水平方向優先級機器人 (`delivering_pod`) 佔該方向總數的比例。
            6.  `vertical_priority_ratio`: 垂直方向優先級機器人 (`delivering_pod`) 佔該方向總數的比例。
            7.  `avg_horizontal_wait_time / 50.0`: 水平方向機器人的平均等待時間，歸一化。
            8.  `avg_vertical_wait_time / 50.0`: 垂直方向機器人的平均等待時間，歸一化。
        * **設計理由 (基於 Q&A 分析)：** 旨在捕獲做出反應式決策所需的**本地核心信息**：當前信號狀態與持續時間、雙向交通負載、任務緊急性（通過優先級比例體現）、擁堵程度（通過平均等待時間體現）。歸一化有助於神經網絡學習穩定性。
        * **充分性與潛在不足：** 對於本地、反應式決策可能是合理的起點。但**缺乏全局信息**（如下游路況、機器人最終目的地）和**預測能力**。聚合信息（如平均等待時間）可能丟失極端個例信息（儘管有防鎖死補充）。狀態中僅區分了 `delivering_pod` 優先級。其有效性需通過實驗結果評估。
    * **動作空間設計 (Action Space - 3 個動作):**
        1.  保持當前信號狀態 (Stay)。
        2.  切換到允許水平方向通行 (Switch to Horizontal)。
        3.  切換到允許垂直方向通行 (Switch to Vertical)。
    * **獎勵函數設計 (Reward Function):**
        * **目標：** 引導智能體學習期望的行為，如減少等待、節省能源、提高通行效率。
        * **組成與權重 (根據 `get_reward` 分析)：**
            * `等待時間減少 (Wait Time Reduction)`: `((prev_h_wait + prev_v_wait) - (curr_h_wait + curr_v_wait)) / 2.0` (隱含權重約 **+0.5** per unit time saved)。
            * `信號切換懲罰 (Switch Penalty)`: 若發生切換，獎勵 `-2.0`；否則為 `0`。
            * `能源消耗懲罰 (Energy Penalty)`: `-0.1 * energy_consumption` (權重 **-0.1** per unit energy)。`energy_consumption` 是上一個狀態轉移區間內，通過該交叉口的機器人**在該交叉口逗留期間的累計能耗**。
            * `停止-啟動懲罰 (Stop-Go Penalty)`: `-0.5 * stop_and_go_count` (權重 **-0.5** per stop-go event)。計算的是上一個區間內通過路口的機器人的停止啟動次數。
            * `機器人通過獎勵 (Passing Reward)`: `+1.0 * robots_passed` (權重 **+1.0** per robot passed)。獎勵上一個區間內成功通過交叉口的機器人數量。
        * **權重來源 (基於 Q&A 分析)：** 這些數值（-2.0, -0.1, -0.5, +1.0 等）**極不可能是理論推導得出**。它們幾乎總是通過**反覆的實驗和啟發式調整 (heuristic tuning)** 來確定的。設計者觀察智能體行為和最終 KPI 來調整權重，以平衡各個目標。
        * **獎勵衝突解決：** 通過**加權求和**的方式隱式解決。智能體的目標是最大化總獎勵值。它會在學習中自動權衡，例如，如果切換信號帶來的高等待時間減少和通過獎勵足以抵消切換懲罰，它就會學會切換。
    * **神經網絡架構 (Neural Network Architecture):**
        * 類型：全連接網絡 (MLP - Multi-Layer Perceptron)。
        * 輸入層：8 個神經元 (對應 8 維狀態)。
        * 隱藏層：2 個隱藏層，每層 24 個神經元，使用 ReLU 激活函數。
        * 輸出層：3 個神經元 (對應 3 個動作的 Q 值)。
    * **模型訓練與保存 (Training & Saving):**
        * **獎勵計算時機：** 潛在地在**每個 tick** 發生狀態轉移 (s, a, s') 後，計算獎勵 r。
        * **經驗存儲：** (s, a, r, s') 元組存儲於經驗回放池 (Replay Buffer)。
        * **學習/訓練時機：** 以固定的時間間隔進行，例如**每 64 個 tick**，從回放池中採樣一個批次 (batch size 32) 的數據，用於訓練 Q 網絡。
        * **目標網絡更新：** 每 1000 個 tick，將 Q 網絡的權重複製到目標網絡 (Target Network)。
        * **自動保存：** 每 5000 個 tick 自動保存當前模型到 `models/{model_name}_{tick}.pth`。
        * 支持暫停和繼續訓練。
    * **模型加載 (Loading):**
        * 支持通過指定完整路徑或指定 tick 數來加載預訓練模型。
    * **運行模式 (Modes):**
        * **訓練模式 (Training Mode):** 進行探索 (ε-greedy)，收集經驗，並按計劃更新網絡。
        * **推理模式 (Inference/Evaluation Mode):** 關閉探索 (ε=0)，僅利用已訓練的網絡權重做出最優決策，不進行網絡更新。
        * 內置整合了防鎖死機制和最小綠燈時間限制。
        * 包含單向交通優化邏輯。
    * **訓練進程監控 (Monitoring Learning Progress - 基於 Q&A 分析):**
        * **目前缺乏**直接的數值指標追蹤，如**損失函數值 (Loss)** 或 **Q 值估計 (Q-value)** 的變化趨勢。
        * 主要依賴**間接觀察仿真過程中的宏觀 KPI** (如 NetLogo UI 上的等待時間、能耗；或模擬結束後生成的報告) 隨時間 (ticks) 的變化。
        * 輔助依賴**基本的訓練過程日誌**（訓練開始、目標網絡更新、記憶庫大小/探索率打印）來確認訓練在進行。

4.  **神經進化強化學習(NERL)控制器 (Neuroevolution RL Controller)**
    * **原理：** 結合了神經網絡（作為策略表示）和進化算法（作為優化方法）。維護一個神經網絡種群 (population)，通過模擬自然選擇和遺傳操作（交叉、變異）來逐步進化出性能更優的網絡（即控制策略）。
    * **特性：** 不需要像 DQN 那樣計算梯度和反向傳播，對獎勵函數的設計可能不那麼敏感，並且具有良好的並行化潛力。
    * **可調參數：**
        * `min_green_time`, `bias_factor`, `max_wait_threshold`, `model_name` (同 DQN)。
        * `population_size`: 種群中的個體（網絡）數量 (預設 40)。
        * `elite_size`: 每代直接保留到下一代的最佳個體數量 (預設 8)。
        * `tournament_size`: 錦標賽選擇中每次比較的個體數量 (預設 4)。
        * `crossover_rate`: 應用交叉操作的概率 (預設 0.7)。
        * `mutation_rate`: 應用變異操作的概率 (預設 0.15)。
        * `mutation_strength`: 變異操作中擾動的強度 (預設 0.1)。
        * `evolution_interval`: 執行一次完整進化操作（評估、選擇、交叉、變異）的時間間隔 (預設 **15 ticks**)。
    * **運作方式：** 與 DQN 類似，中央 `IntersectionManager` 為每個交叉口調用 NERL 控制器的 `get_direction` 方法。在訓練模式下，系統會輪流使用種群中的不同個體網絡進行決策，並收集其表現數據。在評估模式下，則始終使用當前已知的最佳個體網絡。進化過程作用於整個種群，評估基於所有交叉口在評估窗口內的總體性能。
    * **狀態空間設計 (State Space):** 與 DQN 控制器**相同** (8 維向量)。
    * **動作空間設計 (Action Space):** 與 DQN 控制器**相同** (3 個動作)。
    * **適應度函數設計 (Fitness Function - 基於 Q&A 分析):**
        * **核心概念：** NERL 的適應度 (Fitness) 用於評估種群中每個個體（神經網絡）的好壞，以指導進化選擇。
        * **計算方式：**
            1.  在**訓練模式**下，每個個體會輪流控制系統（所有交叉口）一段時間（由內部邏輯決定，並在 `evolution_interval` 內完成對所有個體的評估）。
            2.  當某個體控制時，每次發生狀態轉移，系統會使用與 DQN **相同的 `get_reward()` 方法**計算一個**即時獎勵 (instantaneous reward)**。
            3.  控制器會記錄每個個體在過去的 `evolution_interval` (15 ticks) 內**累積獲得的總即時獎勵 (total_reward)** 以及**成功計算獎勵的次數 (episode_length)**。
            4.  在 `evolution_interval` 結束時，觸發 `_evolve()` 方法。
            5.  每個個體的**適應度分數 (Fitness Score)** 被計算為：`Fitness = total_reward / episode_length`。
        * **本質：** 適應度是該個體在最近的評估窗口 (15 ticks) 內，其控制的所有交叉口決策所獲得的**平均即時獎勵**。它不是評估長期目標完成度，而是評估近期內的平均表現。
        * **`evolution_interval = 15` 的考慮：** 這個間隔相對較短。其設定可能是為了在動態變化的環境中獲得更及時的性能反饋，但也可能不足以完全捕捉某些策略的長期影響。這是一個可能需要根據具體問題進行調整的超參數。
    * **神經網絡架構 (Neural Network Architecture):**
        * 類型：可進化的網絡 (EvolvableNetwork)，通常也是 MLP 結構。
        * 輸入層：8 個神經元。
        * 隱藏層：1 個隱藏層，24 個神經元，ReLU 激活。
        * 輸出層：3 個神經元 (對應 3 個動作的值，但這裡不一定是 Q 值，而是直接的動作偏好或概率)。
    * **進化機制 (Evolutionary Mechanisms):**
        * **精英保留 (Elitism):** 保留上一代適應度最高的 `elite_size` 個個體直接進入下一代。
        * **選擇 (Selection):** 採用錦標賽選擇 (`tournament_size`)，隨機選取若干個體，其中適應度最高的被選為父代。
        * **交叉 (Crossover):** 以 `crossover_rate` 的概率，對選中的父代應用均勻交叉，混合其網絡權重產生子代。
        * **變異 (Mutation):** 以 `mutation_rate` 的概率，對子代的網絡權重添加高斯噪聲（強度由 `mutation_strength` 控制）。
    * **模型訓練與保存 (Training & Saving):**
        * **進化時機：** 每 `evolution_interval` (15 ticks) 進行一次種群評估和進化（選擇、交叉、變異）。
        * **自動保存：** 每 5000 個 tick 保存當前種群中適應度最佳的個體網絡到 `models/{model_name}_{tick}.pth`。
    * **模型加載 (Loading):**
        * 支持通過指定 tick 數加載之前保存的最佳個體模型。
    * **運行模式 (Modes):**
        * **訓練模式 (Training Mode):** 持續進行種群評估和進化。
        * **評估模式 (Evaluation Mode):** 僅使用當前已知的最佳個體網絡進行決策，不進行進化操作。
    * **訓練進程監控 (Monitoring Learning Progress - 基於 Q&A 分析):**
        * 主要指標是**每一代（進化間隔結束時）打印出的最佳個體適應度 (best_fitness)**。觀察此值是否隨進化代數（每 15 ticks 一代）穩定**提升**是判斷進化是否有效的關鍵。
        * **目前缺乏**對種群**平均適應度**的追蹤和報告。
        * 輔助依賴**間接觀察仿真過程中的宏觀 KPI** 隨時間（或進化代數）的變化來判斷最佳策略的實際效果。
        * 包含一些基本的進化過程日誌（進化嘗試、新最佳個體發現等）。

---

### 模擬環境

* **可視化前端：** 基於 **NetLogo** 的 2D 可視化界面，用於展示倉庫佈局、機器人實時位置、狀態、交叉口信號燈等，便於直觀理解系統運行狀況。
* **核心邏輯後端：** 使用 **Python** 實現 RMFS 系統的核心運作邏輯，包括：
    * 世界模型 (倉庫佈局、路徑網絡)
    * 實體對象 (機器人、貨架、工作站)
    * 管理器 (機器人管理、訂單管理、交叉口管理)
    * 交通控制器的具體實現 (Python 類)。
* **能源消耗模型 (Energy Consumption Model - 基於 Q&A 分析):**
    * 採用**基於物理的模型**計算機器人能耗。
    * **區分運動狀態：** 分別計算加速/減速階段和勻速運動階段的能耗。
    * **考慮因素：**
        * 機器人自身質量 (`ROBOT_MASS`)。
        * 負載質量 (`LOAD_MASS`) - **注意：** 默認值可能為 0，需要確保有代碼邏輯在機器人裝卸貨物時動態更新此值才能生效。
        * 摩擦力 (通過 `FRICTION_COEFFICIENT`)。
        * 慣性力 (加速/減速時，通過 `INERTIA`)。
        * 速度 (影響勻速和加減速能耗)。
    * **假設：** 機器人靜止時不消耗能量（或忽略不計）。
    * **公式簡化：** 可能使用了簡化的物理公式（如平均速度的計算方式、`INERTIA` 的具體物理意義可能需要查閱源碼）。
    * **單位：** 存在單位轉換因子 (7200, 3600)，表明結果可能被轉換為特定單位，如瓦時 (Wh)。
* **訂單處理：** 包含完整的訂單生成（可配置）、分配、以及機器人執行任務（取貨、送貨）的機制。

### 評估框架

* **統一指標收集：** 內建 `MetricsCollector` 等模塊，用於在模擬運行過程中持續收集關鍵性能指標 (KPIs)。
* **多場景測試支持：** 設計上支持在不同配置下（如標準負載、高密度交通、變化的訂單到達率）運行模擬，以評估策略的魯棒性。
* **綜合評估維度：**
    * **能源效率：** 如總能耗、平均每任務能耗。
    * **訂單處理效率：** 如訂單完成時間（平均、最大）、吞吐量（單位時間完成訂單數）、機器人利用率。
    * **交通流量指標：** 如平均等待時間、最大等待時間、停止-啟動次數、交叉口通過量。
* **公平比較機制 (基於 Q&A 分析):**
    * **環境一致性：** 所有控制器在相同的仿真環境、倉庫佈局、初始條件下運行。
    * **指標標準化：** 使用同一套 KPI 及其計算方法。
    * **獨立參數配置：** 允許為每種控制器設置其特定的參數或加載其專用的預訓練模型。
    * **用戶責任：** 為了確保比較真正公平，進行比較的**用戶需要負責**為基於規則的控制器（Time-based, Queue-based）進行合理的參數調優，並為基於學習的控制器（DQN, NERL）選擇具有可比性、已充分訓練的模型。框架本身提供比較的工具，但不自動執行跨策略的參數優化。

## 系統架構

```
RMFS/
├── ai/                      # 人工智能與控制器模塊
│   ├── controllers/         # 各種交通控制器實現
│   │   ├── time_based_controller.py
│   │   ├── queue_based_controller.py
│   │   ├── dqn_controller.py
│   │   └── nerl_controller.py
│   ├── deep_q_network.py    # DQN 神經網絡模型
│   ├── evolvable_network.py # NERL 可進化網絡模型
│   └── traffic_controller.py # 控制器抽象基類
├── world/                   # 倉庫世界模型與實體
│   ├── entities/            # 實體類 (Robot, Pod, Station...)
│   ├── managers/            # 管理器類 (IntersectionManager, RobotManager...)
│   └── warehouse.py         # 倉庫整體表示
├── evaluation/              # 評估與數據收集框架
│   ├── metrics_collector.py # 指標收集器
│   └── result_analyzer.py   # (或 PerformanceReportGenerator) 結果分析與報告生成
├── data/                    # 輸入/輸出數據 (地圖, 訂單, 結果)
├── models/                  # 保存訓練好的 DQN/NERL 模型
├── lib/                     # 可能包含輔助庫 (如訂單生成器)
│   └── generator/
│       └── warehouse_generator.py # 包含訂單配置的腳本
├── rmfs.nlogo               # NetLogo 可視化界面文件
├── netlogo.py               # Python 與 NetLogo 的交互接口
├── requirements.txt         # Python 依賴庫列表
└── CHANGELOG.md             # 版本變更與問題修復記錄
```

## 使用方法

1.  **安裝依賴：** `pip install -r requirements.txt` (確保 Python 環境配置正確)。
2.  **啟動 NetLogo：** 打開 NetLogo 應用程序並加載 `rmfs.nlogo` 文件。
3.  **初始化 (Setup)：** 在 NetLogo 界面點擊 "Setup" 按鈕，初始化倉庫環境、機器人、訂單等。
4.  **選擇控制器與配置參數：**
    * 使用 NetLogo 界面上的按鈕（如 `Time-Based`, `Queue-Based`, `DQN`, `NERL`）選擇要使用的交通控制器。
    * 調整相應的滑桿或輸入框來設置該控制器的參數（如 `min-green-time`, `bias-factor` 等）。
    * **DQN 模型加載：**
        * 使用 `model-tick` 滑桿選擇要加載的已保存模型對應的 tick 數。
        * 點擊 `DQN(加載模型)` 按鈕來加載。
        * 點擊 `查看模型` 按鈕列出 `models/` 文件夾下可用的 DQN 模型文件。
    * **NERL 控制器使用：**
        * **基礎設置：** 點擊 `NERL控制器` 按鈕選擇 NERL。
        * **加載模型：** 調整 `model-tick` 滑桿選擇 tick，點擊 `NERL(加載模型)` 加載對應的最佳個體模型。
        * **訓練/評估模式切換：**
            * 點擊 `NERL訓練模式開` 進入訓練模式（會進行進化）。
            * 點擊 `NERL訓練模式關` 進入評估模式（僅使用最佳模型，不進化）。
        * **命令行操作 (供參考)：**
            * 設置 NERL: `nl-command "set-nerl-controller"`
            * 加載模型 (例 tick 5000, 評估模式 exploration=0): `nl-command "set-nerl-controller exploration-rate 0 load-model-tick 5000"`
            * 切換模式: `nl-command "set-nerl-training-mode false"` (評估) 或 `true` (訓練)。
            * 查看模型: `nl-command "print list-models-by-type \\"nerl\\""`
        * **推薦評估設置：** 加載一個預訓練好的模型 (`NERL(加載模型)`), 然後確保處於評估模式 (`NERL訓練模式關`)。
5.  **運行模擬：** 點擊 NetLogo 界面上的 "Go" (運行一步) 或 "Go-forever" (連續運行) 按鈕。
6.  **監控進程：**
    * **NetLogo 界面：** 觀察機器人運動、交叉口狀態以及界面上顯示的實時指標（如 tick 數、平均等待時間等）。
    * **控制台輸出：** 查看 Python 控制台輸出，獲取調試信息、訓練日誌（DQN/NERL 訓練進度、最佳適應度等）。
    * **(高級) 訓練監控:** 如前所述，DQN 需觀察宏觀指標變化；NERL 可主要關注控制台輸出的 `best_fitness` 是否提升。

### 訂單配置參數修改

要調整訂單生成的特性（如數量、到達率、複雜度），需要編輯 **`lib/generator/warehouse_generator.py`** 文件。找到其中的 `draw_layout_from_generated_file` 函數，修改對 `config_orders` 函數的調用參數。

**主要配置段落示例 (行號可能略有變動):**

* **普通訂單 (Normal Orders - 約 70-77行):**
    ```python
    config_orders(
        initial_order=20,                  # 模擬開始時的初始訂單數
        total_requested_item=500,          # 倉庫中獨特物品 (SKU) 的總數
        items_orders_class_configuration={"A": 0.6, "B": 0.3, "C": 0.1}, # 訂單中 ABC 類物品的比例
        quantity_range=[1, 12],            # 每個訂單包含的物品數量範圍
        order_cycle_time=100,              # 平均每小時生成的訂單數量 (數值越小越密集)
        order_period_time=5,               # 訂單生成的總時長 (小時)
        order_start_arrival_time=5,        # 訂單從第幾 tick 開始到達
        # ... 其他參數 ...
    )
    ```
* **積壓訂單 (Backlog Orders - 約 84-94行):** (如果系統配置了積壓訂單生成)
    ```python
    config_orders(
        initial_order=50,                  # 初始積壓訂單數量
        # ... 其他參數類似普通訂單 ...
        sim_ver=2,                         # 通常用版本號區分積壓訂單配置
        dev_mode=True                      # 可能用於觸發不同的生成邏輯
    )
    ```

**重要參數說明：**

* `initial_order`: 模擬開始時就存在的訂單數。
* `total_requested_item`: 影響物品種類的多樣性。
* `order_cycle_time`: 控制訂單到達的頻率/密度。
* `order_period_time`: 控制訂單生成的總量（與 `order_cycle_time` 結合）。
* `quantity_range`: 影響單個訂單的複雜度（需要訪問的貨架可能更多）。

**修改後生效步驟 (關鍵！)：**

在修改 `warehouse_generator.py` 中的參數後，**必須**刪除以下由舊配置生成的數據文件，然後重新運行 NetLogo 的 "Setup" 才能使新配置生效：

1.  `data/input/assign_order.csv`
2.  `data/output/generated_order.csv`
3.  `data/input/generated_backlog.csv` (如果使用)
4.  `data/output/items.csv`
5.  `data/output/skus_data.csv`

可選地，如果想完全重置與貨架相關的生成數據：

* `data/output/generated_pod.csv`

刪除文件後，點擊 NetLogo 中的 "Setup" 按鈕，系統會使用新的參數重新生成訂單和相關數據。

### 調試訊息控制系統 (Debug Message Control)

系統提供了一個全局的調試信息級別控制，通過修改 `Robot` 類的靜態屬性 `DEBUG_LEVEL` 來實現。

**文件位置：** `world/entities/robot.py`

```python
class Robot:
    # ... 其他屬性 ...

    # Debug level: 0 = No debug messages
    #              1 = Important messages only (e.g., warnings, training progress)
    #              2 = Verbose (all detailed messages, e.g., movement, intersection logic)
    DEBUG_LEVEL = 1  # <-- 修改這裡的值 (0, 1, or 2)

    # ... 其他方法 ...
```

**調試級別說明：**

* **`DEBUG_LEVEL = 0`**: **無輸出 (No Output)**。適合正式運行或性能測試，避免控制台信息刷屏。
* **`DEBUG_LEVEL = 1`**: **重要信息 (Important Only)**。輸出關鍵的警告、錯誤信息，以及重要的訓練進程節點（如 DQN 訓練批次開始、目標網絡更新、NERL 新最佳個體發現等）。
* **`DEBUG_LEVEL = 2`**: **詳細信息 (Verbose)**。輸出所有級別的調試信息，包括機器人的詳細運動計算、路徑規劃步驟、交叉路口狀態判斷等細節。適合深入調試特定行為。

**使用方法：**

直接在 `robot.py` 文件中修改 `DEBUG_LEVEL` 的值，保存文件後重新運行 Python 後端（如果需要）或重新 Setup NetLogo 模擬。

**信息類型示例：**

* 訓練信息 (DQN/NERL 訓練狀態，模型保存/加載)
* 機器人運動 (路徑計算，速度調整，到達/離開交叉口)
* 交叉口邏輯 (狀態獲取，決策過程，信號切換)
* 系統級信息 (NetLogo-Python 通信事件，時間同步)

所有調試信息**默認為英文**，便於開發和問題定位。

## 學習範式與模型特性澄清

* **學習範式 (Learning Paradigm):** DQN 和 NERL 都屬於**強化學習 (Reinforcement Learning, RL)** 的範疇。智能體（交通控制器）通過與環境（模擬倉庫）的交互、試錯，並基於獲得的標量獎勵信號（DQN）或適應度評分（NERL）來學習優化其行為策略。這**不是監督學習 (Supervised Learning)**，因為系統不會被告知每個狀態下「正確」的動作是什麼，而是要自己探索發現。
* **黑箱特性 (Black-Box Nature):** 由於 DQN 和 NERL 的核心決策單元是神經網絡，其內部的具體決策過程往往難以像基於規則的系統那樣被直接、清晰地解釋。從這個角度看，它們可以被認為是**黑箱模型**。理解其行為通常需要通過觀察其在不同輸入下的輸出、分析學習曲線以及藉助可視化工具。

## 已知問題與修復

詳細的問題追蹤、功能更新和已修復的 Bug 列表，請參閱項目根目錄下的 **`CHANGELOG.md`** 文件。

