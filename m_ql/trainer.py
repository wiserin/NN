import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import random

class Trainer:
    def __init__(self, model, memory, target_model, batch_size=32, gamma=0.99, lr=0.001):
        self.model = model
        self.memory = memory
        self.target_model = target_model
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_frequency = 1000
        self.steps_done = 0

        self.loss_fn = nn.MSELoss()

        self.target_model.load_state_dict(self.model.state_dict())  # Инициализация целевой модели
        self.target_model.eval()  # Устанавливаем целевую модель в режим "оценки"

    def train_step(self):
        """Один шаг обучения."""
        # Выбор мини-батча
        # batch = self.memory.sample(list(self.batch_size))
        batch_size = min(self.batch_size, len(self.memory))
        batch = random.sample(list(self.memory), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Преобразование данных в тензоры
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Текущие Q-значения
        q_values = self.model(states)[0]
        # q_values = self.model(states)[0].gather(1, actions)

        # Максимальные Q-значения для следующего состояния с учетом маски
        with torch.no_grad():
            next_q_values = self.target_model(next_states)[0]
            mask = next_states.view(next_states.size(0), -1) == 0  # Маска доступных ходов
            masked_next_q_values = torch.where(mask, next_q_values, torch.tensor(float('-inf')))
            max_next_q_values = masked_next_q_values.max(1, keepdim=True)[0]

        # Целевые Q-значения
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Расчёт ошибки
        loss = self.loss_fn(q_values, target_q_values)

        # Обновление весов
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Увеличиваем количество шагов
        self.steps_done += 1
        
        # Обновляем целевую модель, если достигли нужного количества шагов
        if self.steps_done % self.update_frequency == 0:
            self.update_target_model()  # Синхронизируем веса целевой модели
    
    def update_target_model(self):
        """Обновляем целевую модель, копируя веса из основной модели."""
        self.target_model.load_state_dict(self.model.state_dict()) 
