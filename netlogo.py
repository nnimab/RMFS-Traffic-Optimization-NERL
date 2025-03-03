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

        # Check Robot debug level before printing
        from world.entities.robot import Robot
        if Robot.DEBUG_LEVEL > 1:
            print("before tick", warehouse._tick)

        # Update each object with the current warehouse context

        # Perform a simulation tick
        warehouse.tick()

        # Generate results after the tick
        next_result = warehouse.generateResult()
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(warehouse, config_dictionary_file)
        return [next_result, warehouse.total_energy, len(warehouse.job_queue), warehouse.stop_and_go,
                warehouse.total_turning, warehouse._tick]
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
def set_queue_based_controller(min_green_time=30, bias_factor=1.5):
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
def set_dqn_controller(exploration_rate=0.2, load_model_tick=None):
    """
    設置DQN控制器
    
    Args:
        exploration_rate (float): 探索率，控制隨機選擇動作的概率
        load_model_tick (int, optional): 加載特定時間點保存的模型（如5000,10000,20000）
    
    Returns:
        bool: 成功返回True，失敗返回False
    """
    result = set_traffic_controller("dqn", exploration_rate=exploration_rate)
    
    # 如果設置成功且指定了模型，嘗試加載模型
    if result and load_model_tick is not None:
        try:
            with open('netlogo.state', 'rb') as file:
                warehouse: Warehouse = pickle.load(file)
                
            # 嘗試加載特定ticks的模型
            if hasattr(warehouse.intersection_manager, 'controller'):
                load_success = warehouse.intersection_manager.controller.load_model(tick=load_model_tick)
                
                # 保存更新後的狀態
                with open('netlogo.state', 'wb') as file:
                    pickle.dump(warehouse, file)
                    
                print(f"DQN model loading {'successful' if load_success else 'failed'} for tick {load_model_tick}")
                return load_success
        except Exception as e:
            print(f"Error when loading model: {e}")
            traceback.print_exc()
            return False
    
    return result


# 列出可用的模型函數
def list_available_models():
    """
    列出models目錄中所有可用的模型
    
    Returns:
        list: 可用模型列表
    """
    try:
        # 確保models目錄存在
        if not os.path.exists("models"):
            return []
            
        # 獲取所有.pth文件
        model_files = [f for f in os.listdir("models") if f.endswith(".pth")]
        return model_files
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


if __name__ == "__main__":
    result = setup()
    for _ in range(10):
        result = tick()
    # with open('result.txt', 'w') as result_file:
    #     result_file.write(str(result))