# 從用戶角度的 RMFS 交通控制系統使用流程

## 初始啟動系統

當您作為用戶啟動系統時，流程如下：

1. **打開 NetLogo 界面**
   - 載入 `rmfs.nlogo` 模型文件

2. **設置環境**
   - 點擊 "Setup Py" 按鈕安裝必要的依賴包
   - 系統會詢問是否安裝 TensorFlow (用於 DQN 模式)

3. **選擇交通控制方法**
   - 從下拉菜單選擇交通控制策略：
     - TIME_BASED (基於時間的控制)
     - QUEUE_BASED (基於隊列的控制)
     - DQN (深度強化學習控制)

4. **初始化模擬環境**
   - 點擊 "Setup" 按鈕初始化倉庫環境
   - 系統會創建倉庫佈局、機器人、工作站等

## 運行模擬

點擊 "Go" 或 "Go Forever" 按鈕後，系統會:

1. **交通控制策略的運作方式**：
   
   - **TIME_BASED** (基於時間的控制)：
     - 系統會按照預定的時間間隔，定期切換每個交叉路口的通行方向
     - 例如：每30秒允許東西方向通行，然後切換到南北方向
     - 這是最簡單的策略，不考慮實際交通狀況
   
   - **QUEUE_BASED** (基於隊列的控制)：
     - 系統會計算每個方向等待的機器人數量
     - 優先放行等待機器人數量最多的方向
     - 當某個方向等待機器人數量超過閾值時會觸發方向切換
     - 這種策略會根據實際交通狀況動態調整
   
   - **DQN** (深度強化學習控制)：
     - 系統會根據當前的交通狀態，使用深度強化學習來決定最佳的通行方向
     - 在運行過程中，系統會：
       1. 收集交通狀態數據（等待機器人數量、等待時間等）
       2. 將這些數據輸入 DQN 神經網絡
       3. 神經網絡輸出最佳的通行方向決策
       4. 系統執行決策並觀察結果
       5. 使用結果來更新 DQN 模型（訓練過程）

2. **機器人和訂單處理**：
   - 系統會生成新訂單並為其分配工作站
   - 為每個訂單找到合適的貨架（Pod）
   - 為機器人分配任務（取貨、送貨）
   - 機器人移動時會受到您選擇的交通控制策略的影響

3. **性能數據收集**：
   - 系統自動記錄關鍵性能指標：
     - 總能耗和平均能耗
     - 停止-啟動次數
     - 轉彎次數
     - 訂單完成數量
   - 這些數據會保存到 `results` 目錄下的 CSV 文件中

## DQN 模式的特殊運作

如果您選擇 DQN 交通控制策略：

1. **模型載入**：
   - 系統會嘗試載入之前訓練好的 DQN 模型
   - 如果找不到模型，會創建新模型並從頭開始訓練

2. **訓練過程**：
   - 模擬過程即為訓練過程，每個決策都會影響模型
   - 系統使用「經驗回放」(Experience Replay)技術，存儲交通狀態、行動和獎勵
   - 根據「獎勵函數」評估決策好壞（獎勵基於等待時間減少、通行效率等）
   - 模型會逐漸學習最佳的交通控制策略

3. **探索與利用平衡**：
   - 初期階段，系統會更多地「探索」(Exploration)，嘗試不同的通行方向
   - 隨著訓練進行，會逐漸增加「利用」(Exploitation)，選擇已知的好決策
   - 這種平衡由 ε-greedy 策略控制，ε 值會隨時間遞減

4. **模型保存**：
   - 訓練過程中，系統會定期保存模型
   - 模型保存在 `traffic_control/models` 目錄下
   - 下次啟動時可以繼續從保存點訓練

## 結果分析

完成模擬後：

1. **生成比較報告**：
   - 點擊 "產生比較報告" 按鈕
   - 系統會分析所有已運行策略的結果數據
   - 生成文本報告和視覺化圖表

2. **查看報告**：
   - 報告保存在 `results` 目錄下
   - 包括：
     - 文本報告 (`comparison_report_[timestamp].txt`)：詳細數據和分析
     - 圖表 (`comparison_charts.png`)：視覺化比較不同策略的性能

3. **結果解讀**：
   - 較低的能耗、較少的停止-啟動次數和較少的轉彎次數表示更好的交通控制效率
   - DQN 策略通常在足夠訓練後會表現最佳，但需要較長的訓練時間
   - TIME_BASED 策略最簡單但效率最低
   - QUEUE_BASED 策略通常是一個很好的折衷方案

## 進階使用

1. **調整 DQN 參數**：
   - 您可以在 `dqn_controller.py` 中調整學習率、折扣因子等參數
   - 這些參數會影響 DQN 模型的學習效果和速度

2. **自定義交通控制策略**：
   - 您可以通過繼承 `TrafficControllerBase` 類創建自己的控制策略
   - 需要實現 `update` 和 `decide_direction` 等方法
   - 在 `controller.py` 中註冊新策略，即可在界面中選擇使用

通過這種方式，您可以全面了解和控制RMFS系統中的交通控制，並根據不同的倉庫場景選擇最合適的控制策略。 