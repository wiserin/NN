import torch
import random

class Agent:
    def __init__(self, model, epsilon, epsilon_min, epsilon_decay):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Автоматический выбор устройства
        self.model = model.to(self.device)  # Перемещаем модель на устройство
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        """Выбор действия с использованием e-жадной стратегии."""
        # Преобразование state в плоский список
        flat_state = [cell for row in state for cell in row]
        
        # Выбор случайного действия
        if random.random() < self.epsilon:
            available_moves = [i for i, cell in enumerate(flat_state) if cell == 0]
            return random.choice(available_moves) if available_moves else 0

        # Выбор действия на основе предсказания
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            state_tensor = state_tensor / 2.0
            q_values = self.model(state_tensor)[0]  # Предсказание Q-значений
            mask = torch.tensor(flat_state, dtype=torch.bool).to(self.device) == 0  # Маска доступных ходов

            # Применяем маскирование
            masked_q_values = torch.where(mask, q_values, torch.tensor(float('-inf'), dtype=q_values.dtype, device=self.device))
            action = masked_q_values.argmax().item()
            return action

    def random_move(self, state):
        flat_state = [cell for row in state for cell in row]
        available_moves = [i for i, cell in enumerate(flat_state) if cell == 0]
        return random.choice(available_moves) if available_moves else 0

    # def play(self, state):
    #     flat_state = [cell for row in state for cell in row]
    #     mask = torch.tensor(flat_state, dtype=torch.bool).to(self.device) == 0 
    #     with torch.no_grad():
    #         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
    #         state_tensor = state_tensor / 2.0
    #         q_values = self.model(state_tensor)
    #         masked_q_values = torch.where(mask, q_values, torch.tensor(float('-inf'), dtype=q_values.dtype, device=self.device))
    #         action = masked_q_values.argmax().item()
    #         return action

    def update_epsilon(self):
        """Уменьшает значение epsilon для e-жадной стратегии."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
