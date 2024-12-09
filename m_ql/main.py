import torch
from tqdm import trange
from .model import QModel
from .memory import ReplayMemory
from .agent import Agent
from .trainer import Trainer
from .game import Game 

def main():
    TRAIN_FREQUENCY = 1000 #Частота обучения относительно сыгранных эпизодов
    EPSILON_FREQUENCY = 500 #Частота обновления e
    TARGET_MODEL_FREQUENCY = 30 #Частота обновления целевой модели
    TRAIN_SIZE = 300 #Количество батчей
    EPISODES = 3000 #Общее количество игр
    MEMORY_CAPACITY = 1000 #Максимальная загрузка памяти
    BATCH_SIZE = 32
    EPSILON = 1.0 
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.9
    GAMMA = 0.995
    LR = 0.001

    # Создание объектов
    model = QModel()
    game = Game()
    memory = ReplayMemory(
        MEMORY_CAPACITY
    )
    agent = Agent(
        model,
        EPSILON,
        EPSILON_MIN,
        EPSILON_DECAY
    )

    count = 0

    for episode in trange(EPISODES, desc='Processing'):
        done = False
        data_ready = False
        epsilon_ready = False

        while not done:
            can_win = game.can_win_on_this_turn(1)
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
                reward, done = (50, True)
            elif game.check_win(2):
                reward, done = (-50, True)
            elif game.is_full():
                reward, done = (0, True)
            else:
                if can_win:
                    reward, done = (-10, False)
                else:
                    reward, done = (0, False)

            memory.push((state, action, reward, game.board, done))
            count += 1

            if count % TRAIN_FREQUENCY == 0:
                data_ready = True

            if count % EPSILON_FREQUENCY == 0:
                epsilon_ready = True

        if data_ready:

            trainer = Trainer(
                model,
                memory.memory,
                model,
                BATCH_SIZE,
                GAMMA,
                LR
            )
            loss = 0
            for i in range(TRAIN_SIZE):
                loss =+ trainer.train_step()
                if i % TARGET_MODEL_FREQUENCY == 0:
                    trainer.update_target_model()
            # print(f"Loss: {loss / TRAIN_SIZE}")
            model = trainer.model
            agent.model = model
            data_ready = False

        if epsilon_ready:
            # Уменьшение epsilon
            agent.update_epsilon()

        # Перезапуск игры
        game.new_board()

    torch.save(model.state_dict(), "q_model.pth")
    print('Ok')