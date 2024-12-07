import torch
from .model import QModel
from .memory import ReplayMemory
from .agent import Agent
from .trainer import Trainer
from .game import Game 

def main():
    # Создание объектов
    model = QModel()
    memory = ReplayMemory(capacity=1000)
    agent = Agent(model)
    trainer = Trainer(model, memory.memory, model)

    # Начало игры
    game = Game()

    for episode in range(1000):
        state = game.board
        done = False

        while not done:
            action = agent.select_action(state)

            # Преобразование action в координаты (x, y)
            x, y = divmod(action, 5)
            if game.board[x][y] != 0:
                continue  # Пропускаем недопустимый ход

            game.move(1, (x, y))  # Ход игрока 1
            reward, done = (1, True) if game.check_win(1) else (0, False)

            # Сохранение в память
            memory.push((state, action, reward, game.board, done))

            # Обучение
            trainer.train_step()

            # Смена состояния
            state = game.board

        # Уменьшение epsilon
        agent.update_epsilon()

        # Перезапуск игры
        game.new_board()

    torch.save(model, "q_model_full.pth")