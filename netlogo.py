import pickle
import os
import traceback
from typing import List

from lib.generator.warehouse_generator import *
from pip._internal import main as pipmain
from lib.file import *
from world.warehouse import Warehouse

def setup():
    try:
        # Initialize the simulation warehouse
        assignment_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        if os.path.exists(assignment_path):
            os.remove(assignment_path)
        warehouse = Warehouse()
        
        # Populate the warehouse with objects and connections
        draw_layout(warehouse)
        # print(warehouse.intersection_manager.intersections[0].intersection_coordinate)

        # Generate initial results
        next_result = warehouse.generateResult()
        
        warehouse.initWarehouse();

        # Save the warehouse state for future ticks
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(warehouse, config_dictionary_file)

        return next_result

    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."


def tick():
    try:
        # print("========tick========")

        # Load the simulation state
        with open('netlogo.state', 'rb') as file:
            warehouse: Warehouse = pickle.load(file)

        print("before tick", warehouse._tick)

        # Update each object with the current warehouse context

        # Perform a simulation tick
        warehouse.tick()

        # Generate results after the tick
        next_result = warehouse.generateResult()
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(warehouse, config_dictionary_file)
        return [next_result, warehouse.total_energy, len(warehouse.job_queue), warehouse.stop_and_go,
                warehouse.total_turning]
    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."


def setup_py():
    def install_package(package_name):
        """Install a Python package using pip."""
        pipmain(['install', package_name])

    # List of packages to install
    packages = ["networkx", "matplotlib"]

    # Install each package
    for package in packages:
        install_package(package)


# 新增設置交通控制器的函數
def set_traffic_controller(controller_type, **kwargs):
    """
    從NetLogo設置交通控制器類型
    
    Args:
        controller_type (str): 控制器類型，例如 "time_based", "queue_based", "dqn", "nerl"
        **kwargs: 控制器需要的額外參數
    
    Returns:
        bool: 成功返回True，失敗返回False
    """
    try:
        # 加載模擬狀態
        with open('netlogo.state', 'rb') as file:
            warehouse: Warehouse = pickle.load(file)
        
        # 設置交通控制器
        success = warehouse.set_traffic_controller(controller_type, **kwargs)
        
        # 保存模擬狀態
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(warehouse, config_dictionary_file)
            
        print(f"交通控制器已設置為: {controller_type}")
        return success
    except Exception as e:
        # 打印完整堆疊信息
        traceback.print_exc()
        return False


# 設置時間基控制器的快捷函數
def set_time_based_controller(horizontal_time=70, vertical_time=30):
    """
    設置時間基控制器
    
    Args:
        horizontal_time (int): 水平方向綠燈時間
        vertical_time (int): 垂直方向綠燈時間
    
    Returns:
        bool: 成功返回True，失敗返回False
    """
    return set_traffic_controller("time_based", 
                               horizontal_green_time=horizontal_time, 
                               vertical_green_time=vertical_time)


# 設置隊列基控制器的快捷函數
def set_queue_based_controller(min_green_time=10, bias_factor=1.5):
    """
    設置隊列基控制器
    
    Args:
        min_green_time (int): 最小綠燈時間
        bias_factor (float): 水平方向偏好因子
    
    Returns:
        bool: 成功返回True，失敗返回False
    """
    return set_traffic_controller("queue_based", 
                               min_green_time=min_green_time, 
                               bias_factor=bias_factor)


# 設置DQN控制器的快捷函數
def set_dqn_controller(exploration_rate=0.2):
    """
    設置DQN控制器
    
    Args:
        exploration_rate (float): 探索率，控制隨機選擇動作的概率
    
    Returns:
        bool: 成功返回True，失敗返回False
    """
    return set_traffic_controller("dqn", exploration_rate=exploration_rate)


if __name__ == "__main__":
    result = setup()
    for _ in range(10):
        result = tick()
    # with open('result.txt', 'w') as result_file:
    #     result_file.write(str(result))