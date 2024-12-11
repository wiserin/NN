import torch
import random
from tqdm import trange
from .model import QModel
from .memory import ReplayMemory
from .agent import Agent
from .trainer import Trainer
from .game import Game

def main(player, enemy):
    """
    Основная функция первичного обучения.
    1 - Крестик
    2 - Нолик
    """

    TRAIN_FREQUENCY = 5000  # Частота обучения относительно сыгранных эпизодов
    EPSILON_FREQUENCY = 500  # Частота обновления e отнолительно эпизодов
    TARGET_MODEL_FREQUENCY = 50  # Частота обновления целевой модели
    TRAIN_SIZE = 400  # Количество батчей
    EPISODES = 50000  # Общее количество игр
    MEMORY_CAPACITY = 9000  # Максимальная загрузка памяти
    BATCH_SIZE = 32
    EPSILON = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.9
    GAMMA = 0.99
    LR = 0.0005

    # Определяем устройство (CPU или GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Создание объектов
    model = QModel().to(device)  # Перенос модели на устройство
    game = Game()
    memory = ReplayMemory(MEMORY_CAPACITY)
    agent = Agent(
        model,
        EPSILON,
        EPSILON_MIN,
        EPSILON_DECAY
    )

    count = 0
    winning = 0
    loss = 0
    draw = 0
    epoch = 0
    stat_count = 0
    for episode in range(EPISODES,):
        done = False
        data_ready = False
        epsilon_ready = False

        while not done:
            can_win = game.can_win_on_this_turn(player)
            state = game.board
            action = agent.select_action(state)

            # Преобразование action в координаты (x, y)
            x, y = divmod(action, 5)
            if game.board[x][y] != 0:
                continue  # Пропускаем недопустимый ход
            
            # Если противник О, то первым ходит Х, затем рандомно О
            if player == 1:
                game.move(player, (x, y))  # Ход Х

                # Ход О
                O_action = agent.random_move(game.board)
                x, y = divmod(O_action, 5)
                game.move(enemy, (x, y))

            # если противник Х, то наоборот 
            elif player == 2:
                # Рандомный ход Х
                X_action = agent.random_move(game.board)
                x_, y_ = divmod(X_action, 5)
                game.move(enemy, (x_, y_))

                game.move(player, (x, y))  # Ход О

            if game.check_win(player):
                reward, done = (70, True)
                winning += 1
            elif game.check_win(enemy):
                reward, done = (-70, True)
                loss += 1
            elif game.is_full():
                reward, done = (0, True)
                draw += 1
            else:
                if can_win:
                    reward, done = (-5, False)
                else:
                    reward, done = (0, False)

            reward += random.uniform(-0.1, 0.1)

            memory.push((state, action, reward, game.board, done))
            count += 1

            if count % TRAIN_FREQUENCY == 0:
                data_ready = True

            if count % EPSILON_FREQUENCY == 0:
                epsilon_ready = True

        stat_count += 1


        if data_ready:
            trainer = Trainer(
                model,
                memory.memory,
                model,
                BATCH_SIZE,
                GAMMA,
                LR
            )
            for i in range(TRAIN_SIZE,):
                trainer.train_step()
                if i % TARGET_MODEL_FREQUENCY == 0:
                    trainer.update_target_model()
            
            # Обновление модели
            model = trainer.model
            agent.model = model
            data_ready = False

            # Вывод статистики
            epoch += 1
            print(f'Epoch: {epoch}\nWinning: {winning/stat_count*100:.2f}%\nLoss: {loss/stat_count*100:.2f}%\nDraw: {draw/stat_count*100:.2f}%\n\n')
            winning, loss, draw, stat_count = 0, 0, 0, 0

        if epsilon_ready:
            # Уменьшение epsilon
            agent.update_epsilon()
            if agent.epsilon < 0.1:
                print('Epsilon')

        # Перезапуск игры
        game.new_board()

    torch.save(model.state_dict(), f"q_model_{player}.pth")
    print('Ok')
