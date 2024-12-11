import torch
from .memory import ReplayMemory
from .trainer import Trainer
from .model import QModel
from .game import Game
import random

def main():
    TRAIN_FREQUENCY = 900  # Частота обучения относительно сыгранных игр
    EPSILON_FREQUENCY = 200  # Частота обновления e отнолительно игр
    TARGET_MODEL_FREQUENCY = 50  # Частота обновления целевой модели
    TRAIN_SIZE = 300  # Количество батчей
    EPISODES = 50000  # Общее количество игр
    MEMORY_CAPACITY = 4000  # Максимальная загрузка памяти
    BATCH_SIZE = 32
    EPSILON = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.999
    GAMMA = 0.99
    LR = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_x = QModel().to(device)
    model_x.load_state_dict(torch.load('q_model_1.pth'))

    model_o = QModel().to(device)
    model_o.load_state_dict(torch.load('q_model_2.pth'))

    memory_x = ReplayMemory(MEMORY_CAPACITY)
    memory_o = ReplayMemory(MEMORY_CAPACITY)

    game = Game()
    epsilon = EPSILON

    win_x = 0
    win_o = 0
    draw = 0
    stat_count = 0
    epoch = 0

    for episode in range(EPISODES):
        winner = play_game(
            model_x,
            model_o,
            game,
            device,
            memory_x,
            memory_o,
            epsilon
        )

        if winner == 1:
            win_x += 1
        elif winner == 2:
            win_o += 1
        elif winner == 0:
            draw += 1

        stat_count += 1

        if episode % TRAIN_FREQUENCY == 0:
            trainer_x = Trainer(
                model_x,
                memory_x.memory,
                model_x,
                BATCH_SIZE,
                GAMMA,
                LR
            )

            trainer_o = Trainer(
                model_o,
                memory_o.memory,
                model_o,
                BATCH_SIZE,
                GAMMA,
                LR
            )

            for i in range(TRAIN_SIZE):
                trainer_x.train_step()
                trainer_o.train_step()

                if i % TARGET_MODEL_FREQUENCY == 0:
                    trainer_x.update_target_model()
                    trainer_o.update_target_model()

            model_x = trainer_x.model
            model_o = trainer_o.model

            epoch += 1
            print(f'Epoch: {epoch}\nX won: {win_x/stat_count*100:.2f}%\nO won: {win_o/stat_count*100:.2f}%\nDraw: {draw/stat_count*100:.2f}%')

        # Уменьшение ε
        if episode % EPSILON_FREQUENCY == 0:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


    torch.save(model_x.state_dict(), f"q_model_upd_x.pth")
    torch.save(model_o.state_dict(), f"q_model_upd_o.pth")
    print('Training completed')


def play_game(model_x, model_o, game, device, memory_x, memory_o, epsilon):
    done = False
    game.new_board()
    state = game.board

    while not done:
        reward_winner = 50
        reward_loser = -50
        reward_draw = 5
        reward_step = -4

        for agent, memory, enemy_memory, current_player in [(model_x, memory_x, memory_o, 1), (model_o, memory_o, memory_x, 2)]:
            if done:
                break  # Если игра завершена после хода одного из игроков

            # Преобразуем состояние в плоский список
            flat_state = [cell for row in state for cell in row]

            # Выбор действия
            if random.random() < epsilon:
                available_moves = [i for i, cell in enumerate(flat_state) if cell == 0]
                action = random.choice(available_moves) if available_moves else 0
            else:
                with torch.no_grad():  # Выключаем отслеживание градиентов
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 2.0
                    q_values = agent(state_tensor)[0]
                    mask = torch.tensor(flat_state, dtype=torch.bool).to(device) == 0
                    masked_q_values = torch.where(mask, q_values, torch.tensor(float('-inf'), device=device))
                    action = masked_q_values.argmax().item()

            # Преобразование действия
            x, y = divmod(action, 5)

            # Совершаем ход
            game.move(current_player, (x, y))

            # Проверка результата игры
            if game.check_win(current_player):
                memory.push((state, action, reward_winner, game.board, True))
                enemy_memory.update_reward(reward_loser, True)
                return current_player

            elif game.is_full():
                memory.push((state, action, reward_draw, game.board, True))
                enemy_memory.update_reward(reward_draw, True)
                return 0

            else:
                # Сохранение опыта
                memory.push((state, action, reward_step, game.board, False))

            # Обновление состояния
            state = game.board


