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

    def update_reward(self, reward, done):
        # Достаём последний ход из памяти
        last_transition = self.memory[-1]  # Последний добавленный элемент (state, action, reward, next_state, done)

        # Обновляем награду
        updated_transition = (
            last_transition[0],  # state
            last_transition[1],  # action
            reward,          # обновляем reward
            last_transition[3],  # next_state
            done,  # done
        )

        # Заменяем последний элемент
        self.memory[-1] = updated_transition
