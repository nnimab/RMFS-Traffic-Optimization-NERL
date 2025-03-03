from ai.traffic_controller import TrafficController
from collections import namedtuple

# 將 GridPosition 定義為可序列化的類
class GridPosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __eq__(self, other):
        if not isinstance(other, GridPosition):
            return False
        return self.x == other.x and self.y == other.y
        
    def __hash__(self):
        return hash((self.x, self.y))
        
    def __repr__(self):
        return f"GridPosition(x={self.x}, y={self.y})"
        
    # 添加序列化支持
    def __reduce__(self):
        return (GridPosition, (self.x, self.y))

class QueueBasedController(TrafficController):
    """
    基於隊列長度和任務優先級的智能交通控制器
    
    主要功能：
    1. 根據等待機器人數量和任務優先級動態調整交通方向
    2. 為水平方向設置可調偏好因子，反映pods分佈特性
    3. 設置最小綠燈時間，避免頻繁切換
    4. 當只有一個方向有機器人時自動給予通行權
    5. 根據機器人任務狀態（運送中、返回等）分配不同權重
    6. 綠波協調機制，確保機器人能連續通過多個交叉路口
    """
    
    def __init__(self, min_green_time=30, bias_factor=1.5, green_wave_enabled=True, **kwargs):
        """
        初始化基於隊列的控制器
        
        Args:
            min_green_time (int): 最小綠燈時間，避免頻繁切換
            bias_factor (float): 水平方向偏好因子，反映pods分佈特性
            green_wave_enabled (bool): 是否啟用綠波協調
            **kwargs: 其他參數
        """
        super().__init__(controller_name="隊列基控制器")
        self.min_green_time = min_green_time  # 最小綠燈時間
        self.bias_factor = bias_factor  # 水平方向偏好因子
        self.last_change = {}  # 記錄每個交叉路口上次更改的時間
        self.current_direction = {}  # 當前方向
        self.green_wave_enabled = green_wave_enabled  # 是否啟用綠波
        self.adjacent_intersections = {}  # 存儲交叉路口相鄰關係
        self.green_wave_influence = 5.0  # 綠波影響因子 (從3.0增加到5.0)
        self.green_wave_propagation_time = 15  # 綠波傳播時間 (從10增加到15)
        self.robots_tracking = {}  # 跟踪機器人移動路徑
        self.robot_movement_history = {}  # 機器人穿越交叉路口的軌跡
        
        # 機器人狀態權重映射表
        self.status_weights = {
            "delivering_pod": 2.0,   # 運送中的機器人優先級最高
            "returning_pod": 1.5,    # 返回中的機器人優先級較高
            "taking_pod": 1.2,       # 取貨中的機器人優先級中等
            "idle": 1.0              # 空閒機器人優先級最低
        }
    
    def initialize_adjacent_intersections(self, warehouse):
        """
        初始化交叉路口相鄰關係表
        
        Args:
            warehouse: 倉庫對象
        """
        if not hasattr(warehouse, 'intersection_manager') or not warehouse.intersection_manager.intersections:
            print("Warning: No intersection manager or intersections found")
            self.green_wave_enabled = False  # 禁用綠波功能
            return False
            
        # 處理intersections可能是列表或字典的情況
        if isinstance(warehouse.intersection_manager.intersections, dict):
            all_intersections = warehouse.intersection_manager.intersections.values()
            print(f"Initializing green wave with {len(warehouse.intersection_manager.intersections)} intersections (dict)")
        else:  # 假設是列表
            all_intersections = warehouse.intersection_manager.intersections
            print(f"Initializing green wave with {len(warehouse.intersection_manager.intersections)} intersections (list)")
        
        # 建立交叉路口的相鄰關係表
        missing_grid_position_count = 0
        created_connections = 0
        
        for intersection in all_intersections:
            self.adjacent_intersections[intersection.id] = {
                "Horizontal": [],
                "Vertical": []
            }
            
            # 檢查此交叉路口是否有grid_position
            has_grid_position = hasattr(intersection, 'grid_position')
            
            # 如果沒有grid_position，嘗試從coordinate或x,y屬性創建
            if not has_grid_position:
                if hasattr(intersection, 'coordinate'):
                    # 從NetLogoCoordinate創建grid_position
                    intersection.grid_position = GridPosition(intersection.coordinate.x, intersection.coordinate.y)
                    has_grid_position = True
                    print(f"Created grid_position for {intersection.id} from coordinate attribute")
                elif hasattr(intersection, 'pos_x') and hasattr(intersection, 'pos_y'):
                    # 從pos_x和pos_y創建grid_position
                    intersection.grid_position = GridPosition(intersection.pos_x, intersection.pos_y)
                    has_grid_position = True
                    print(f"Created grid_position for {intersection.id} from pos_x/pos_y attributes")
                else:
                    missing_grid_position_count += 1
            
            # 找出水平和垂直方向的相鄰交叉路口
            for other in all_intersections:
                if intersection.id != other.id:
                    # 檢查兩個交叉路口是否都有grid_position
                    other_has_grid_position = hasattr(other, 'grid_position')
                    
                    # 為other創建grid_position（如果可能）
                    if not other_has_grid_position:
                        if hasattr(other, 'coordinate'):
                            other.grid_position = GridPosition(other.coordinate.x, other.coordinate.y)
                            other_has_grid_position = True
                        elif hasattr(other, 'pos_x') and hasattr(other, 'pos_y'):
                            other.grid_position = GridPosition(other.pos_x, other.pos_y)
                            other_has_grid_position = True
                    
                    # 如果任一交叉路口沒有grid_position，則跳過
                    if not has_grid_position or not other_has_grid_position:
                        continue
                        
                    # 如果X座標相同且Y座標距離較近，認為是垂直方向的相鄰交叉路口
                    # 將距離閾值從10增加到15，以便識別更多相鄰交叉路口
                    if intersection.grid_position.x == other.grid_position.x and abs(intersection.grid_position.y - other.grid_position.y) <= 15:
                        self.adjacent_intersections[intersection.id]["Vertical"].append({
                            "id": other.id,
                            "distance": abs(intersection.grid_position.y - other.grid_position.y)
                        })
                        created_connections += 1
                    
                    # 如果Y座標相同且X座標距離較近，認為是水平方向的相鄰交叉路口
                    # 將距離閾值從10增加到15，以便識別更多相鄰交叉路口
                    if intersection.grid_position.y == other.grid_position.y and abs(intersection.grid_position.x - other.grid_position.x) <= 15:
                        self.adjacent_intersections[intersection.id]["Horizontal"].append({
                            "id": other.id,
                            "distance": abs(intersection.grid_position.x - other.grid_position.x)
                        })
                        created_connections += 1
        
        # 按距離排序相鄰交叉路口
        for intersection_id in self.adjacent_intersections:
            for direction in ["Horizontal", "Vertical"]:
                self.adjacent_intersections[intersection_id][direction].sort(key=lambda x: x["distance"])
                
        # 輸出統計信息
        total_intersections = len(self.adjacent_intersections)
        total_connections = sum(len(data["Horizontal"]) + len(data["Vertical"]) for data in self.adjacent_intersections.values())
        
        print(f"Green wave initialized with {total_intersections} intersections and {total_connections} connections")
        print(f"Missing grid_position for {missing_grid_position_count} intersections")
        print(f"Created {created_connections} directional connections between intersections")
        
        # 檢查是否有足夠的連接建立
        if total_connections == 0:
            print("Warning: No connections established between intersections. Green wave disabled.")
            self.green_wave_enabled = False
            return False
            
        return True
    
    def track_robot_movement(self, intersection, robot, tick):
        """
        跟踪機器人穿越交叉路口的軌跡，用於綠波協調
        
        Args:
            intersection: 交叉路口對象
            robot: 機器人對象
            tick: 當前時間刻
        """
        # 確保機器人有id屬性
        if not hasattr(robot, 'id'):
            print(f"Warning: Robot has no id attribute")
            return
            
        robot_id = robot.id
        
        try:
            # 如果這是機器人第一次出現在跟踪記錄中，初始化其記錄
            if robot_id not in self.robot_movement_history:
                self.robot_movement_history[robot_id] = {
                    'last_intersection': None,
                    'direction': None,
                    'timestamp': 0
                }
            
            # 獲取機器人的移動方向
            direction = self._get_robot_direction(robot)
            
            # 如果機器人換了交叉路口
            if self.robot_movement_history[robot_id]['last_intersection'] != intersection.id:
                # 更新機器人的交叉路口和方向記錄
                self.robot_movement_history[robot_id] = {
                    'last_intersection': intersection.id,
                    'direction': direction,
                    'timestamp': tick
                }
        except Exception as e:
            print(f"Error tracking robot movement: {e}")
    
    def _get_robot_direction(self, robot):
        """
        根據機器人的朝向確定其移動方向
        
        Args:
            robot: 機器人對象
            
        Returns:
            str: "Horizontal" 或 "Vertical"
        """
        # 根據機器人朝向確定方向
        if robot.heading in [0, 180]:  # 朝北或朝南，垂直移動
            return "Vertical"
        else:  # 朝東或朝西，水平移動
            return "Horizontal"
    
    def calculate_green_wave_influence(self, intersection_id, direction, tick, warehouse):
        """
        計算綠波對當前交叉路口的影響
        
        Args:
            intersection_id: 交叉路口ID
            direction: 要計算的方向 ("Horizontal" 或 "Vertical")
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            float: 綠波影響係數
        """
        # 如果沒有啟用綠波功能或相鄰交叉路口表為空，則無影響
        if not self.green_wave_enabled or not self.adjacent_intersections:
            return 0.0
            
        # 如果交叉路口未在相鄰表中註冊，則無影響
        if intersection_id not in self.adjacent_intersections:
            return 0.0
            
        # 獲取相鄰交叉路口
        try:
            adjacent_intersections = self.adjacent_intersections[intersection_id][direction]
        except Exception as e:
            print(f"Error accessing adjacent intersections for {intersection_id}, direction {direction}: {e}")
            return 0.0
            
        # 如果沒有相鄰交叉路口，則無影響
        if not adjacent_intersections:
            return 0.0
            
        # 計算綠波影響
        green_wave_score = 0.0
        
        for adj_data in adjacent_intersections:
            try:
                adj_id = adj_data["id"]
                distance = adj_data["distance"]
                
                # 獲取相鄰交叉路口的當前允許方向
                adj_intersection = warehouse.intersection_manager.intersection_id_to_intersection.get(adj_id)
                if adj_intersection is None:
                    continue
                    
                adj_direction = adj_intersection.allowed_direction
                
                # 如果相鄰交叉路口的方向與當前方向一致，形成綠波
                if adj_direction == direction:
                    # 獲取上次變化時間
                    if adj_id in self.current_direction:
                        last_change = self.last_change.get(adj_id, 0)
                        time_since_change = tick - last_change
                        
                        # 如果在綠波傳播時間窗口內，加入影響
                        if time_since_change < self.green_wave_propagation_time:
                            # 修改距離衰減計算方式，降低距離對影響的抑制
                            # 原來: influence = self.green_wave_influence * (1.0 - time_since_change / self.green_wave_propagation_time) / max(1, distance)
                            # 現在: 使用 sqrt(distance) 代替 distance，並添加0.5的基礎值，確保即使距離較遠也有一定影響
                            time_factor = 1.0 - time_since_change / self.green_wave_propagation_time
                            distance_factor = 1.0 / max(1, (distance ** 0.5))  # 使用平方根降低距離影響
                            base_influence = 0.5  # 基礎影響值，確保即使距離較遠也有影響
                            influence = self.green_wave_influence * time_factor * (distance_factor + base_influence)
                            green_wave_score += influence
                            
                            # 記錄詳細的綠波影響信息用於調試
                            if influence > 0.5:
                                print(f"Green wave from {adj_id} to {intersection_id}: influence={influence:.2f}, distance={distance}, time_since_change={time_since_change}")
            except Exception as e:
                print(f"Error calculating green wave influence from {adj_id}: {e}")
                continue
                
        return green_wave_score
    
    def get_direction(self, intersection, tick, warehouse):
        """
        根據交叉路口狀態和綠波影響，決定當前交叉路口的通行方向
        
        Args:
            intersection: 交叉路口對象
            tick: 當前時間刻
            warehouse: 倉庫對象
            
        Returns:
            str: 允許通行的方向，"Horizontal" 或 "Vertical"
        """
        intersection_id = intersection.id
        
        # 首次運行時初始化相鄰交叉路口表
        if not self.adjacent_intersections and self.green_wave_enabled:
            try:
                # 如果初始化失敗，initialize_adjacent_intersections 會將 self.green_wave_enabled 設為 False
                initialization_success = self.initialize_adjacent_intersections(warehouse)
                if not initialization_success:
                    print(f"Green wave initialization failed for intersection {intersection_id}. Falling back to basic traffic control.")
            except Exception as e:
                print(f"Error initializing adjacent intersections: {e}")
                # 如果初始化失敗，禁用綠波功能
                self.green_wave_enabled = False
        
        # 獲取水平和垂直方向的機器人數量
        try:
            horizontal_robots = intersection.horizontal_robots.values() if isinstance(intersection.horizontal_robots, dict) else intersection.horizontal_robots
            vertical_robots = intersection.vertical_robots.values() if isinstance(intersection.vertical_robots, dict) else intersection.vertical_robots
        except Exception as e:
            print(f"Error accessing robot lists: {e}")
            horizontal_robots = []
            vertical_robots = []
        
        # 跟踪所有機器人
        if self.green_wave_enabled:
            try:
                for robot in list(horizontal_robots) + list(vertical_robots):
                    if robot is not None:  # 確保機器人對象不為None
                        self.track_robot_movement(intersection, robot, tick)
            except Exception as e:
                print(f"Error tracking robots: {e}")
        
        # 如果只有一個方向有機器人，直接給予通行權
        if len(horizontal_robots) > 0 and len(vertical_robots) == 0:
            return "Horizontal"
        elif len(vertical_robots) > 0 and len(horizontal_robots) == 0:
            return "Vertical"
        
        # 初始化方向權重
        horizontal_weight = 0
        vertical_weight = 0
        
        # 計算水平方向的權重（考慮機器人狀態）
        for robot in horizontal_robots:
            try:
                # 獲取機器人狀態權重，如果狀態不在字典中則使用默認值1.0
                status_weight = self.status_weights.get(getattr(robot, 'current_state', None), 1.0)
                horizontal_weight += status_weight
            except Exception as e:
                print(f"Error calculating horizontal weight: {e}")
                horizontal_weight += 1.0  # 使用默認權重
        
        # 計算垂直方向的權重（考慮機器人狀態）
        for robot in vertical_robots:
            try:
                status_weight = self.status_weights.get(getattr(robot, 'current_state', None), 1.0)
                vertical_weight += status_weight
            except Exception as e:
                print(f"Error calculating vertical weight: {e}")
                vertical_weight += 1.0  # 使用默認權重
        
        # 考慮方向偏好 - 水平方向有優勢
        horizontal_weight = horizontal_weight * self.bias_factor
        
        # 加入綠波影響因子
        if self.green_wave_enabled:
            try:
                horizontal_wave = self.calculate_green_wave_influence(intersection_id, "Horizontal", tick, warehouse)
                vertical_wave = self.calculate_green_wave_influence(intersection_id, "Vertical", tick, warehouse)
                
                horizontal_weight += horizontal_wave
                vertical_weight += vertical_wave
                
                # 調試輸出
                if horizontal_wave > 0 or vertical_wave > 0:
                    print(f"Green Wave influence - Intersection: {intersection_id}, H: {horizontal_wave:.2f}, V: {vertical_wave:.2f}")
            except Exception as e:
                print(f"Error calculating green wave influence: {e}")
                # 在計算綠波影響時出錯，但不影響基本交通控制邏輯
        
        # 獲取當前方向，如果沒有則設為None
        if intersection_id not in self.current_direction:
            self.current_direction[intersection_id] = intersection.allowed_direction
            self.last_change[intersection_id] = tick
        
        current_dir = self.current_direction[intersection_id]
        last_change_tick = self.last_change[intersection_id]
        
        # 檢查是否滿足最小綠燈時間
        if current_dir is not None and tick - last_change_tick < self.min_green_time:
            return current_dir
        
        # 根據權重決定方向
        if horizontal_weight > vertical_weight and current_dir != "Horizontal":
            self.current_direction[intersection_id] = "Horizontal"
            self.last_change[intersection_id] = tick
            return "Horizontal"
        elif vertical_weight > horizontal_weight and current_dir != "Vertical":
            self.current_direction[intersection_id] = "Vertical"
            self.last_change[intersection_id] = tick
            return "Vertical"
        
        # 如果權重相同並且當前沒有設定方向，預設選擇水平方向
        if current_dir is None:
            self.current_direction[intersection_id] = "Horizontal"
            self.last_change[intersection_id] = tick
            return "Horizontal"
            
        # 否則保持當前方向
        return current_dir 