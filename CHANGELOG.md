# 更新日誌 (Changelog)

本文件包含項目的所有重要更改。

## v0.1.3 (2024-03-02)

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

## v0.1.2 (2024-03-02)

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

## v0.1.1 (2024-03-02)

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