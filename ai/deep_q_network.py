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


class DeepQNetwork:
    def __init__(self, state_size, action_size, model_name):
        self.state_size = state_size
        self.action_size = action_size
        self.model_name = model_name

    def act(self, state):
        # 返回預設動作，例如 0
        return 0

    def remember(self, state, action, reward, next_state, done):
        pass

    def replay(self, batch_size):
        pass

    def save_model(self, model_name, tick):
        pass
