import torch
from .model import QModel
from .memory import ReplayMemory
from .agent import Agent
from .trainer import Trainer
from .game import Game 

def main():
    # Создание объектов
    train_frequency = 1000
    model = QModel()
    memory = ReplayMemory(capacity=1000)
    agent = Agent(model)
    trainer = Trainer(model, memory.memory, model)

    # Начало игры
    game = Game()

    count = 0

    for episode in range(10000):
        done = False
        data_ready = False

        while not done:
            state = game.board
            action = agent.select_action(state)

            # Преобразование action в координаты (x, y)
            x, y = divmod(action, 5)
            if game.board[x][y] != 0:
                continue  # Пропускаем недопустимый ход

            game.move(1, (x, y))  # Ход игрока 1

            O_action = agent.random_move(game.board)
            x, y = divmod(O_action, 5)
            game.move(2, (x, y))

            if game.check_win(1):
                reward, done = (1, True)
            elif game.check_win(2):
                reward, done = (-1, True)
            elif game.is_full():
                reward, done = (0.5, True)
            else:
                reward, done = (-0.1, False)

            memory.push((state, action, reward, game.board, done))
            count += 1

            if count % train_frequency == 0:
                data_ready = True 

        if data_ready:
            print('Обучаюсь')
            for _ in range(100):
                trainer.train_step()
            # Уменьшение epsilon
            agent.update_epsilon()
            data_ready = False

        # Перезапуск игры
        game.new_board()
        print('Я играю')

    torch.save(model.state_dict(), "q_model.pth")