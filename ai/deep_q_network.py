import os
import random
from collections import deque

from world.entities.object import *
from world.warehouse import *
from lib import *
from lib.types.netlogo_coordinate import *
from lib.types.coordinate import *
from lib.types.heading import *
from lib.types.movement import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """深度Q網絡模型"""
    
    def __init__(self, state_size, action_size):
        """
        初始化Q網絡
        
        Args:
            state_size (int): 狀態空間維度
            action_size (int): 動作空間維度
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x (Tensor): 輸入狀態張量
            
        Returns:
            Tensor: Q值張量
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepQNetwork:
    """深度Q學習網絡管理器"""
    
    def __init__(self, state_size, action_size, model_name="dqn_traffic"):
        """
        初始化深度Q網絡
        
        Args:
            state_size (int): 狀態空間維度
            action_size (int): 動作空間維度
            model_name (str): 模型名稱，用於保存和加載模型
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model_name = model_name
        self.memory = deque(maxlen=2000)  # 經驗回放緩衝區
        
        # 超參數
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰減
        self.learning_rate = 0.001  # 學習率
        
        # 創建Q網絡和目標網絡
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()  # 初始化目標網絡權重
        
        # 優化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()  # 均方誤差損失函數
        
        # 加載模型（如果存在）
        self.load_model()
    
    def update_target_model(self):
        """更新目標網絡權重為主網絡權重"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        將經驗存入記憶庫
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一狀態
            done: 是否結束
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        根據當前狀態選擇動作（ε-greedy策略）
        
        Args:
            state: 當前狀態
            
        Returns:
            int: 選擇的動作索引
        """
        # 轉換狀態為張量
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # 探索：隨機選擇動作
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # 利用：選擇具有最高Q值的動作
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        
        return torch.argmax(action_values[0]).item()
    
    def replay(self, batch_size=32):
        """
        從記憶庫中抽樣進行經驗回放和網絡訓練
        
        Args:
            batch_size (int): 批次大小
        """
        # 如果記憶庫中樣本不足，直接返回
        if len(self.memory) < batch_size:
            return
        
        # 導入 Robot 類以使用 DEBUG_LEVEL
        from world.entities.robot import Robot
        
        # 從記憶庫中隨機抽樣
        minibatch = random.sample(self.memory, batch_size)
        
        if Robot.DEBUG_LEVEL > 0:
            print(f"[Training] Training DQN with batch size {batch_size}, memory size {len(self.memory)}")
        
        for state, action, reward, next_state, done in minibatch:
            # 轉換為張量
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            action = torch.tensor(action)
            reward = torch.tensor(reward)
            done = torch.tensor(done, dtype=torch.float32)
            
            # 計算目標Q值
            self.model.eval()
            with torch.no_grad():
                target = self.model(state)
                next_q_values = self.target_model(next_state)
                max_next_q = torch.max(next_q_values)
                
                # Q學習更新規則
                target[action] = reward + (1 - done) * self.gamma * max_next_q
            
            # 預測Q值
            self.model.train()
            predicted = self.model(state)
            
            # 計算損失並更新網絡
            loss = self.loss_fn(predicted[action], target[action])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 衰減探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load_model(self, model_path=None):
        """
        加載已保存的模型
        
        Args:
            model_path (str, optional): 完整模型路徑，如果未提供則使用默認模型
            
        Returns:
            bool: 是否成功加載模型
        """
        if not model_path:
            model_path = f"models/{self.model_name}.pth"
            
        if os.path.isfile(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"Model loaded: {model_path}")
                self.update_target_model()
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Model file not found: {model_path}")
            return False
    
    def save_model(self, model_name=None, tick=None):
        """
        保存模型
        
        Args:
            model_name (str, optional): 模型名稱，默認為None使用self.model_name
            tick (int, optional): 當前時間節點，用於標識保存的模型版本
        """
        # 創建models目錄（如果不存在）
        os.makedirs("models", exist_ok=True)
        
        # 確定保存路徑
        save_name = model_name if model_name else self.model_name
        if tick is not None:
            save_name = f"{save_name}_{tick}"
        
        save_path = f"models/{save_name}.pth"
        
        try:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved: {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
