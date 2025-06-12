import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class PerformanceReportGenerator:
    """
    綜合性能報告生成器，用於分析和記錄模擬結束後的關鍵績效指標。
    這個報告將幫助比較不同交通控制器的效能。
    """
    
    def __init__(self, warehouse=None, result_dir=None, controller_name=None):
        """
        初始化報告生成器
        
        Args:
            warehouse: Warehouse對象，如果提供則直接從中獲取數據
            result_dir (str): 模擬結果目錄的路徑，僅當warehouse為None時使用
            controller_name (str): 使用的控制器名稱，僅當warehouse為None時使用
        """
        self.warehouse = warehouse
        self.result_dir = result_dir
        
        # 更改控制器名稱的獲取邏輯，確保不使用"none"
        if controller_name:
            self.controller_name = controller_name
        elif warehouse and warehouse.current_controller and warehouse.current_controller != "none":
            self.controller_name = warehouse.current_controller
        else:
            self.controller_name = "unknown"
            
        # 如果控制器名稱為"none"，則替換為更具描述性的名稱
        if self.controller_name == "none":
            self.controller_name = "default"
            
        self.date_string = warehouse.landscape.current_date_string if warehouse else os.path.basename(result_dir) if result_dir else datetime.now().strftime("%Y-%m-%d-%H%M%S")
        
        self.result_files = {}
        if not warehouse and result_dir:
            self.result_files = self._get_result_files()
            
        # 時間序列數據收集屬性
        self.time_series_data = {
            "ticks": [],
            "total_energy_consumption": [],
            "avg_robot_utilization": [],
            "avg_intersection_traffic": [],
            "avg_intersection_wait_time": [],
            "completed_orders_count": [],
            "avg_intersection_congestion": []
        }
        self.last_collection_tick = 0
        self.collection_interval = 10  # 每10個tick收集一次數據
    
    def _get_result_files(self):
        """獲取結果文件路徑，僅當warehouse為None時使用"""
        files = {}
        if os.path.exists(os.path.join(self.result_dir, "order-finished.csv")):
            files["orders"] = os.path.join(self.result_dir, "order-finished.csv")
        if os.path.exists(os.path.join(self.result_dir, "intersection-energy-consumption.csv")):
            files["intersections"] = os.path.join(self.result_dir, "intersection-energy-consumption.csv")
        return files
    
    def collect_time_series_data(self, current_tick=None):
        """
        收集當前時刻的KPI時間序列數據
        
        Args:
            current_tick (float, optional): 當前模擬時間，如果為None則使用warehouse的_tick
            
        Returns:
            bool: 如果數據被收集則返回True，否則返回False
        """
        if not self.warehouse:
            print("Error: Cannot collect time series data without a warehouse object")
            return False
            
        current_tick = current_tick or self.warehouse._tick
        
        # 檢查是否需要收集數據（根據收集間隔）
        if current_tick - self.last_collection_tick < self.collection_interval:
            return False
            
        # 更新上次收集時間
        self.last_collection_tick = current_tick
        
        # 計算當前的KPI值
        kpis = {}
        self._generate_kpis_from_warehouse(kpis)
        
        # 儲存時間序列數據
        self.time_series_data["ticks"].append(current_tick)
        self.time_series_data["total_energy_consumption"].append(kpis["total_energy_consumption"])
        self.time_series_data["avg_robot_utilization"].append(kpis["avg_robot_utilization"])
        self.time_series_data["avg_intersection_traffic"].append(kpis["avg_intersection_traffic"])
        self.time_series_data["avg_intersection_wait_time"].append(kpis["avg_intersection_wait_time"])
        self.time_series_data["completed_orders_count"].append(kpis["completed_orders_count"])
        self.time_series_data["avg_intersection_congestion"].append(kpis["avg_intersection_congestion"])
        
        return True
    
    def save_time_series_data(self):
        """
        將收集的時間序列數據保存到JSON文件中
        
        Returns:
            str: 保存的文件路徑
        """
        # 確保結果目錄存在
        time_series_dir = os.path.join("result", "time_series")
        if not os.path.exists(time_series_dir):
            os.makedirs(time_series_dir)
            
        # 確保控制器名稱不是'none'
        controller_name = self.controller_name
        if controller_name == "none" or controller_name == "unknown":
            if self.warehouse and self.warehouse.current_controller and self.warehouse.current_controller != "none":
                controller_name = self.warehouse.current_controller
            else:
                controller_name = "default"
        
        # 創建文件名
        file_name = f"time_series_{controller_name}_{self.date_string}.json"
        file_path = os.path.join(time_series_dir, file_name)
        
        # 保存數據
        with open(file_path, 'w') as f:
            json.dump(self.time_series_data, f, indent=2)
            
        print(f"Time series data saved to {file_path}")
        return file_path
    
    def generate_charts(self):
        """
        根據時間序列數據生成圖表
        
        Returns:
            list: 圖表文件路徑列表
        """
        if len(self.time_series_data["ticks"]) == 0:
            print("No time series data available for chart generation")
            return []
            
        # 確保圖表目錄存在
        charts_dir = os.path.join("result", "charts")
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
            
        # 確保控制器名稱不是'none'
        controller_name = self.controller_name
        if controller_name == "none" or controller_name == "unknown":
            if self.warehouse and self.warehouse.current_controller and self.warehouse.current_controller != "none":
                controller_name = self.warehouse.current_controller
            else:
                controller_name = "default"
                
        # 臨時保存舊值，以便在方法結束時恢復
        old_controller_name = self.controller_name
        self.controller_name = controller_name
            
        chart_files = []
        
        # 生成單一指標圖表
        metrics = [
            ("total_energy_consumption", "Total Energy Consumption", "Energy"),
            ("avg_robot_utilization", "Average Robot Utilization (%)", "Utilization Rate (%)"),
            ("avg_intersection_traffic", "Average Intersection Traffic", "Traffic Rate"),
            ("avg_intersection_wait_time", "Average Intersection Waiting Time", "Wait Time (ticks)"),
            ("completed_orders_count", "Completed Orders Count", "Orders"),
            ("avg_intersection_congestion", "Average Intersection Congestion", "Congestion Level")
        ]
        
        for metric_key, title, ylabel in metrics:
            chart_path = self._create_single_metric_chart(charts_dir, metric_key, title, ylabel)
            if chart_path:
                chart_files.append(chart_path)
                
        # 生成多指標比較圖表 (機器人利用率、交叉口流量和等待時間)
        comparison_chart_path = self._create_comparison_chart(charts_dir)
        if comparison_chart_path:
            chart_files.append(comparison_chart_path)
            
        # 新增：生成 NERL 適應度圖表 (如果適用)
        nerl_fitness_chart_path = self._create_nerl_fitness_chart(charts_dir)
        if nerl_fitness_chart_path:
            chart_files.append(nerl_fitness_chart_path)
            
        # 恢復原始控制器名稱
        self.controller_name = old_controller_name
            
        return chart_files
        
    def _create_single_metric_chart(self, charts_dir, metric_key, title, ylabel):
        """
        創建單一指標的時間序列圖表
        
        Args:
            charts_dir (str): 圖表保存目錄
            metric_key (str): 指標鍵名
            title (str): 圖表標題
            ylabel (str): Y軸標籤
            
        Returns:
            str: 圖表文件路徑，如果失敗則返回None
        """
        try:
            # 確保控制器名稱不是'none'
            controller_name = self.controller_name
            if controller_name == "none" or controller_name == "unknown":
                controller_name = "default"
                
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_series_data["ticks"], self.time_series_data[metric_key])
            plt.title(f"{title} - {controller_name}")
            plt.xlabel("Simulation Time (ticks)")
            plt.ylabel(ylabel)
            plt.grid(True)
            
            # 添加數據波動和趨勢說明
            if len(self.time_series_data["ticks"]) > 1:
                # 計算簡單統計
                mean_value = np.mean(self.time_series_data[metric_key])
                max_value = np.max(self.time_series_data[metric_key])
                min_value = np.min(self.time_series_data[metric_key])
                
                # 添加統計信息
                plt.axhline(y=mean_value, color='r', linestyle='--', alpha=0.3)
                plt.text(self.time_series_data["ticks"][0], mean_value, 
                        f" Mean: {mean_value:.2f}", va='center')
                
                # 添加趨勢線 (簡單線性回歸)
                x = np.array(self.time_series_data["ticks"])
                y = np.array(self.time_series_data[metric_key])
                coeffs = np.polyfit(x, y, 1)
                poly_line = np.poly1d(coeffs)
                plt.plot(x, poly_line(x), "r--", alpha=0.5)
            
            # 保存圖表
            file_name = f"chart_{metric_key}_{controller_name}_{self.date_string}.png"
            file_path = os.path.join(charts_dir, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Chart generated: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error generating chart for {metric_key}: {e}")
            return None
            
    def _create_comparison_chart(self, charts_dir):
        """
        創建多指標比較圖表
        
        Args:
            charts_dir (str): 圖表保存目錄
            
        Returns:
            str: 圖表文件路徑，如果失敗則返回None
        """
        try:
            # 確保控制器名稱不是'none'
            controller_name = self.controller_name
            if controller_name == "none" or controller_name == "unknown":
                controller_name = "default"
                
            # 創建帶有四個子圖的圖表 (從三個改為四個)
            fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
            fig.suptitle(f"Performance Metrics - {controller_name}", fontsize=16)
            
            # 1. 機器人利用率
            axs[0].plot(self.time_series_data["ticks"], 
                       [v * 100 for v in self.time_series_data["avg_robot_utilization"]], 
                       'b-', linewidth=2)
            axs[0].set_ylabel("Robot Utilization (%)")
            axs[0].set_title("Average Robot Utilization")
            axs[0].grid(True)
            
            # 設定Y軸範圍從0到100
            axs[0].set_ylim(0, 100)
            
            # 繪製平均線
            mean_util = np.mean([v * 100 for v in self.time_series_data["avg_robot_utilization"]])
            axs[0].axhline(y=mean_util, color='r', linestyle='--', alpha=0.5)
            axs[0].text(self.time_series_data["ticks"][0], mean_util, 
                      f" Mean: {mean_util:.2f}%", va='center')
            
            # 2. 交叉口流量
            axs[1].plot(self.time_series_data["ticks"], 
                       self.time_series_data["avg_intersection_traffic"], 
                       'g-', linewidth=2)
            axs[1].set_ylabel("Traffic Rate")
            axs[1].set_title("Average Intersection Traffic")
            axs[1].grid(True)
            
            # 繪製平均線
            mean_traffic = np.mean(self.time_series_data["avg_intersection_traffic"])
            axs[1].axhline(y=mean_traffic, color='r', linestyle='--', alpha=0.5)
            axs[1].text(self.time_series_data["ticks"][0], mean_traffic, 
                      f" Mean: {mean_traffic:.4f}", va='center')
            
            # 3. 交叉口等待時間
            axs[2].plot(self.time_series_data["ticks"], 
                       self.time_series_data["avg_intersection_wait_time"], 
                       'r-', linewidth=2)
            axs[2].set_ylabel("Wait Time (ticks)")
            axs[2].set_title("Average Intersection Waiting Time")
            axs[2].grid(True)
            
            # 繪製平均線
            mean_wait = np.mean(self.time_series_data["avg_intersection_wait_time"])
            axs[2].axhline(y=mean_wait, color='r', linestyle='--', alpha=0.5)
            axs[2].text(self.time_series_data["ticks"][0], mean_wait, 
                      f" Mean: {mean_wait:.2f}", va='center')
            
            # 4. 新增：訂單數量
            axs[3].plot(self.time_series_data["ticks"], 
                       self.time_series_data["completed_orders_count"], 
                       'purple', linewidth=2, marker='o')
            axs[3].set_ylabel("Orders Count")
            axs[3].set_title("Completed Orders Over Time")
            axs[3].grid(True)
            axs[3].set_xlabel("Simulation Time (ticks)")
            
            # 如果有足夠的數據點，添加訂單增長率趨勢線
            if len(self.time_series_data["ticks"]) > 2:
                # 計算訂單增長趨勢
                x = np.array(self.time_series_data["ticks"])
                y = np.array(self.time_series_data["completed_orders_count"])
                # 只使用非零值進行趨勢線擬合
                non_zero_indices = y > 0
                if np.any(non_zero_indices):
                    x_valid = x[non_zero_indices]
                    y_valid = y[non_zero_indices]
                    if len(x_valid) > 1:  # 至少需要兩個點來擬合線
                        coeffs = np.polyfit(x_valid, y_valid, 1)
                        poly_line = np.poly1d(coeffs)
                        axs[3].plot(x_valid, poly_line(x_valid), "r--", alpha=0.7, 
                                  label=f"Growth trend: {coeffs[0]:.4f} orders/tick")
                        axs[3].legend(loc='upper left')
            
            # 調整子圖之間的間距
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # 保存圖表
            file_name = f"chart_comparison_{controller_name}_{self.date_string}.png"
            file_path = os.path.join(charts_dir, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Comparison chart generated: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error generating comparison chart: {e}")
            return None
            
    def _create_nerl_fitness_chart(self, charts_dir):
        """
        創建 NERL Best 和 Average Fitness 的時間序列圖表
        
        Args:
            charts_dir (str): 圖表保存目錄
            
        Returns:
            str: 圖表文件路徑，如果數據不存在或繪製失敗則返回None
        """
        # 檢查是否為 NERL 控制器，並且是否有歷史數據
        nerl_controller = None
        if self.warehouse and self.warehouse.current_controller == "nerl":
            controller = self.warehouse.intersection_manager.controllers.get('nerl')
            if controller and hasattr(controller, 'best_fitness_history') and hasattr(controller, 'average_fitness_history'):
                if controller.best_fitness_history and controller.average_fitness_history:
                    nerl_controller = controller
        
        if not nerl_controller:
            # print("NERL controller or fitness history not found, skipping fitness chart.")
            return None # 不是 NERL 或沒有數據，不生成圖表
            
        try:
            best_history = nerl_controller.best_fitness_history
            avg_history = nerl_controller.average_fitness_history
            generations = list(range(len(best_history)))
            
            if not generations:
                print("NERL fitness history is empty, skipping fitness chart.")
                return None
                
            plt.figure(figsize=(10, 6))
            plt.plot(generations, best_history, label='Best Fitness', marker='o')
            plt.plot(generations, avg_history, label='Average Fitness', marker='x')
            plt.title(f"NERL Fitness Evolution - {self.controller_name}")
            plt.xlabel("Generation")
            plt.ylabel("Fitness Score")
            plt.legend()
            plt.grid(True)
            
            # 添加趨勢線 (如果數據點足夠)
            if len(generations) > 1:
                # Best Fitness Trend
                x = np.array(generations)
                y_best = np.array(best_history)
                coeffs_best = np.polyfit(x, y_best, 1)
                poly_line_best = np.poly1d(coeffs_best)
                plt.plot(x, poly_line_best(x), "b--", alpha=0.5, label=f'Best Trend ({coeffs_best[0]:.2f})')
                
                # Average Fitness Trend
                y_avg = np.array(avg_history)
                coeffs_avg = np.polyfit(x, y_avg, 1)
                poly_line_avg = np.poly1d(coeffs_avg)
                plt.plot(x, poly_line_avg(x), "r--", alpha=0.5, label=f'Avg Trend ({coeffs_avg[0]:.2f})')
                
                plt.legend() # 更新圖例以包含趨勢線

            # 保存圖表
            file_name = f"chart_nerl_fitness_{self.controller_name}_{self.date_string}.png"
            file_path = os.path.join(charts_dir, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"NERL Fitness chart generated: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error generating NERL fitness chart: {e}")
            import traceback
            traceback.print_exc() # 打印詳細錯誤
            return None
            
    def generate_report(self):
        """生成綜合性能報告"""
        # 計算KPIs
        kpis = {}
        
        # 記錄控制器名稱
        kpis["controller_name"] = self.controller_name
        
        if self.warehouse:
            # 直接從Warehouse對象獲取數據
            self._generate_kpis_from_warehouse(kpis)
            
            # 保存時間序列數據
            if len(self.time_series_data["ticks"]) > 0:
                self.save_time_series_data()
                # 生成圖表
                self.generate_charts()
        else:
            # 從結果文件中獲取數據
            if not self.result_files.get("orders") or not self.result_files.get("intersections"):
                print(f"Error: Required result files not found in directory {self.result_dir}")
                return False
                
            self._generate_kpis_from_files(kpis)
        
        # 生成報告文件
        self._save_report(kpis)
        
        return kpis
    
    def _generate_kpis_from_warehouse(self, kpis):
        """直接從Warehouse對象生成KPIs"""
        # 1. 總能量消耗
        kpis["total_energy_consumption"] = self.warehouse.total_energy
        
        # 2. 平均訂單處理時間和其他訂單相關指標
        processing_times = []
        total_orders = 0
        max_completion_time = 0
        
        for order in self.warehouse.order_manager.orders:
            if order.order_complete_time > 0:  # 只計算已完成的訂單
                processing_time = order.order_complete_time - order.process_start_time
                processing_times.append(processing_time)
                max_completion_time = max(max_completion_time, order.order_complete_time)
                total_orders += 1
        
        if processing_times:
            kpis["avg_order_processing_time"] = sum(processing_times) / len(processing_times)
        else:
            kpis["avg_order_processing_time"] = 0
        
        # 新增: 已完成訂單數量
        kpis["completed_orders_count"] = total_orders
        
        # 8. 完成全部訂單所需的總時間
        kpis["total_completion_time"] = max_completion_time if max_completion_time > 0 else self.warehouse._tick
        
        # 3. 機器人利用率 - 改進的計算方法
        total_active_time = 0
        total_robot_time = 0
        current_tick = self.warehouse._tick
        
        for robot in self.warehouse.robot_manager.getAllRobots():
            # 計算每個機器人的總時間
            robot_active_time = robot.total_active_time
            
            # 如果機器人當前處於活動狀態，加上當前活動的時間
            if robot.current_state != 'idle' and robot.last_state_change_time > 0:
                robot_active_time += current_tick - robot.last_state_change_time
                
            total_active_time += max(0, robot_active_time)  # 確保單個機器人的活動時間不為負
            total_robot_time += max(1, current_tick)  # 確保總時間至少為1，避免除以零
        
        # 計算平均利用率，並確保結果在0到1之間
        kpis["avg_robot_utilization"] = max(0, min(1, total_active_time / total_robot_time if total_robot_time > 0 else 0))
        
        # 5. 停止-啟動次數
        kpis["total_stop_and_go"] = self.warehouse.stop_and_go
        
        # 4, 6, 7. 交叉口相關指標
        intersections = self.warehouse.intersection_manager.getAllIntersections()
        wait_times = []
        traffic_rates = []
        congestion_levels = []
        
        for intersection in intersections:
            # 使用改進的方法計算平均等待時間
            wait_time = intersection.getAverageWaitingTime()
            if wait_time > 0:
                wait_times.append(wait_time)
            
            # 使用改進的方法計算交叉口流量（單位時間內）
            traffic_rate = intersection.getAverageTrafficRate(current_tick)
            traffic_rates.append(traffic_rate)
            
            # 計算當前擁堵程度（當前在交叉口的機器人數量）
            current_robots = intersection.robotCount()
            congestion_levels.append(current_robots)
        
        # 4. 平均交叉口等待時間
        kpis["avg_intersection_wait_time"] = sum(wait_times) / len(wait_times) if wait_times else 0
        
        # 6. 每個路口的流量
        kpis["avg_intersection_traffic"] = sum(traffic_rates) / len(traffic_rates) if traffic_rates else 0
        kpis["max_intersection_traffic"] = max(traffic_rates) if traffic_rates else 0
        
        # 7. 交叉口擁堵程度
        kpis["avg_intersection_congestion"] = sum(congestion_levels) / len(congestion_levels) if congestion_levels else 0
        kpis["max_intersection_congestion"] = max(congestion_levels) if congestion_levels else 0
    
    def _generate_kpis_from_files(self, kpis):
        """從結果文件生成KPIs"""
        # 讀取數據
        orders_df = pd.read_csv(self.result_files["orders"])
        intersections_df = pd.read_csv(self.result_files["intersections"])
        
        # 1. 總能量消耗
        kpis["total_energy_consumption"] = intersections_df["energy_consumption_intersection"].sum()
        
        # 2. 平均訂單處理時間
        orders_df["processing_time"] = orders_df["order_complete_time"] - orders_df["process_start_time"]
        kpis["avg_order_processing_time"] = orders_df["processing_time"].mean()
        
        # 新增: 已完成訂單數量
        kpis["completed_orders_count"] = len(orders_df)
        
        # 3. 機器人利用率 (基於交叉口數據)
        # 提取所有唯一的機器人
        all_robots = intersections_df["robot_name"].unique()
        
        # 計算每個機器人的總活動時間
        robot_active_times = {}
        for robot in all_robots:
            robot_data = intersections_df[intersections_df["robot_name"] == robot]
            active_time = 0
            for _, row in robot_data.iterrows():
                active_time += row["intersection_finish_time"] - row["intersection_start_time"]
            robot_active_times[robot] = active_time
        
        # 計算平均機器人利用率 (活動時間與總模擬時間比例)
        total_simulation_time = max(1, orders_df["order_complete_time"].max())
        if total_simulation_time > 0:
            utilization_rates = [max(0, min(1, active_time / total_simulation_time)) for active_time in robot_active_times.values()]
            kpis["avg_robot_utilization"] = np.mean(utilization_rates) if utilization_rates else 0
        else:
            kpis["avg_robot_utilization"] = 0
        
        # 4. 平均交叉口等待時間
        intersections_df["wait_time"] = intersections_df["intersection_finish_time"] - intersections_df["intersection_start_time"]
        kpis["avg_intersection_wait_time"] = intersections_df["wait_time"].mean()
        
        # 5. 計算停止-啟動次數 (從能量消耗數據中估計)
        # 這是一個估計值，實際值應該從倉庫對象中獲取
        kpis["total_stop_and_go"] = len(intersections_df[intersections_df["energy_consumption_intersection"] > 0])
        
        # 6. 每個路口的流量
        intersection_traffic = intersections_df.groupby("intersection_id").size()
        kpis["avg_intersection_traffic"] = intersection_traffic.mean()
        kpis["max_intersection_traffic"] = intersection_traffic.max()
        
        # 7. 交叉口擁堵程度 (基於等待機器人數量)
        kpis["avg_intersection_congestion"] = intersections_df["queueing_robot"].mean()
        kpis["max_intersection_congestion"] = intersections_df["queueing_robot"].max()
        
        # 8. 完成全部訂單所需的總時間
        kpis["total_completion_time"] = orders_df["order_complete_time"].max()
    
    def _save_report(self, kpis):
        """保存報告到CSV文件"""
        # 確保報告目錄存在
        reports_dir = os.path.join("result", "reports")
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        # 報告文件路徑
        report_file = os.path.join(reports_dir, "performance_summary.csv")
        file_exists = os.path.exists(report_file)
        
        # 寫入報告
        with open(report_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # 寫入表頭
                writer.writerow(kpis.keys())
            # 寫入值
            writer.writerow(kpis.values())
        
        print(f"Performance report generated: {report_file}")
        
        # 還生成一個人類可讀的文本報告
        self._generate_text_report(kpis)
    
    def _generate_text_report(self, kpis):
        """生成人類可讀的文本報告"""
        # 報告文件路徑
        reports_dir = os.path.join("result", "reports")
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        text_report_file = os.path.join(reports_dir, 
                                      f"report_{self.controller_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(text_report_file, 'w') as f:
            f.write(f"=== 倉庫模擬性能報告 ===\n")
            f.write(f"控制器: {kpis['controller_name']}\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("--- 關鍵績效指標 ---\n")
            f.write(f"1. 總能量消耗: {kpis['total_energy_consumption']:.6f}\n")
            f.write(f"2. 平均訂單處理時間: {kpis['avg_order_processing_time']:.2f} 秒\n")
            f.write(f"3. 機器人平均利用率: {kpis['avg_robot_utilization']*100:.2f}%\n")
            f.write(f"4. 平均交叉口等待時間: {kpis['avg_intersection_wait_time']:.2f} 秒\n")
            f.write(f"5. 總停止-啟動次數 (估計): {kpis['total_stop_and_go']}\n")
            f.write(f"6. 平均交叉口流量: {kpis['avg_intersection_traffic']:.2f} 機器人/交叉口\n")
            f.write(f"7. 最高交叉口流量: {kpis['max_intersection_traffic']} 機器人\n")
            f.write(f"8. 平均交叉口擁堵程度: {kpis['avg_intersection_congestion']:.2f} 機器人/交叉口\n")
            f.write(f"9. 最高交叉口擁堵程度: {kpis['max_intersection_congestion']} 機器人\n")
            f.write(f"10. 完成全部訂單所需的總時間: {kpis['total_completion_time']:.2f} 秒\n")
            f.write(f"11. 已完成訂單數量: {kpis['completed_orders_count']} 個\n")
            f.write(f"12. 耗時幾個ticks完成: {kpis['total_completion_time']:.0f} ticks\n")
        
        print(f"Detailed text report generated: {text_report_file}")

def generate_performance_report_from_warehouse(warehouse):
    """
    直接從Warehouse對象生成性能報告
    
    Args:
        warehouse (Warehouse): 倉庫對象
        
    Returns:
        dict: 包含KPI的字典
    """
    # 創建報告生成器
    report_generator = PerformanceReportGenerator(warehouse=warehouse)
    
    # 生成報告
    kpis = report_generator.generate_report()
    return kpis

def generate_performance_report(result_folder, controller_name):
    """
    為指定的結果文件夾生成性能報告
    
    Args:
        result_folder (str): 結果文件夾路徑
        controller_name (str): 控制器名稱
    
    Returns:
        dict: 計算的KPI值字典
    """
    generator = PerformanceReportGenerator(result_dir=result_folder, controller_name=controller_name)
    return generator.generate_report() 