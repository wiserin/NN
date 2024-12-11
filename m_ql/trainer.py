import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import random

class Trainer:
    def __init__(self, model, gamma, lr):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)  # Перенос модели на устройство
        self.target_model = model.to(self.device)  # Перенос целевой модели на устройство
        self.gamma = gamma
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()

        # Инициализация целевой модели
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def train_step(self, batch): 
        """Один шаг обучения."""
        states, actions, rewards, next_states, dones = zip(*batch)

        # Преобразование данных в тензоры
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(1).to(self.device)
        states = states / 2.0
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = next_states / 2.0
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Текущие Q-значения
        q_values = self.model(states).gather(1, actions)

        # Максимальные Q-значения для следующего состояния с учетом маски
        with torch.no_grad():
            next_q_values = self.target_model(next_states)

            # Маска доступных ходов
            mask = next_states.view(next_states.size(0), -1) == 0

            # Применяем маску, чтобы заменить все недоступные позиции на -inf
            masked_next_q_values = torch.where(
                mask, 
                next_q_values, 
                torch.tensor(float('-inf'), dtype=torch.float32).to(next_q_values.device)
            )

            # Поиск максимальных Q-значений среди доступных ходов
            max_next_q_values = masked_next_q_values.max(1, keepdim=True)[0]

            # Обработка случая, если нет доступных ходов (все -inf)
            max_next_q_values = torch.where(
                mask.any(dim=1, keepdim=True), 
                max_next_q_values, 
                torch.zeros_like(max_next_q_values)
            )

        # Целевые Q-значения
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Расчёт ошибки
        loss = self.loss_fn(q_values, target_q_values)

        # Обновление весов
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # return loss

    def update_target_model(self):
        """Обновляем целевую модель, копируя веса из основной модели."""
        self.target_model.load_state_dict(self.model.state_dict())


