import torch
import random

class Agent:
    def __init__(self, model, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.model = model
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
            state_tensor = torch.tensor(flat_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            q_values = self.model(state_tensor)[0]  # Предсказание Q-значений
            mask = torch.tensor(flat_state) == 0  # Маска доступных ходов

            # Применяем маскирование
            masked_q_values = torch.where(mask, q_values, torch.tensor(float('-inf')))
            probabilities = torch.nn.functional.softmax(masked_q_values, dim=0)
            action = torch.multinomial(probabilities, 1).item()  # Выбираем действие
            return action

    def update_epsilon(self):
        """Уменьшает значение epsilon для e-жадной стратегии."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
