import random
from collections import deque
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        """Добавление перехода (state, action, reward, next_state, done)"""
        self.memory.append(transition)

    def sample(self, batch_size):
        """Случайная выборка"""
        batch_size = min(batch_size, len(self.memory))  # Не больше, чем есть в памяти
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
