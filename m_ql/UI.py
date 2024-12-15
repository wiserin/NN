from tkinter import Tk, ttk
from .game import Game
from .model import QModel
import tkinter as tk
import torch
from tkinter import messagebox

class GameUI:
    def __init__(self):
        self.game = Game()
        self.window = Tk()
        self.window.title("Крестики-нолики 5x5")
        self.buttons = [[None for _ in range(5)] for _ in range(5)]
        self.current_player = 1  # Игрок 1 = 'X', Игрок 2 = 'O'
        self._create_ui()

    def _create_ui(self):
        """Создает элементы пользовательского интерфейса."""
        for row in range(5):
            for col in range(5):
                button = tk.Button(
                    self.window,
                    text=" ",
                    font=("Arial", 24),
                    width=4,
                    height=2,
                    command=lambda r=row, c=col: self._make_move(r, c),
                )
                button.grid(row=row, column=col)
                self.buttons[row][col] = button

        reset_button = tk.Button(
            self.window,
            text="Новая игра",
            font=("Arial", 16),
            command=self._reset_game,
        )
        reset_button.grid(row=5, column=0, columnspan=5)

    def _make_move(self, row, col):
        """Обрабатывает ход игрока."""
        if self.game.board[row][col] != 0:
            return  # Ячейка уже занята

        # Совершаем ход
        self.game.move(self.current_player, (row, col))
        self.buttons[row][col]["text"] = "X" if self.current_player == 1 else "O"

        # Проверка на победу
        if self.game.check_win(self.current_player):
            winner = "Игрок 1 (X)" if self.current_player == 1 else "Игрок 2 (O)"
            messagebox.showinfo("Победа!", f"{winner} выиграл!")
            self._reset_game()
            return

        # Проверка на ничью
        if self.game.is_full():
            messagebox.showinfo("Ничья", "Игра закончилась вничью!")
            self._reset_game()
            return

        # Смена игрока
        self.current_player = 2 if self.current_player == 1 else 1

    def _reset_game(self):
        """Сбрасывает игровое поле и очищает UI."""
        self.game.new_board()
        for row in range(5):
            for col in range(5):
                self.buttons[row][col]["text"] = " "
        self.current_player = 1

    def run(self):
        """Запускает интерфейс."""
        self.window.mainloop()


class GameWithAI_X(GameUI):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = QModel().to(self.device)  # Загружаем модель на CUDA/CPU
        self.model.load_state_dict(torch.load('D:/Programs/Tima/NN/models/second/x/q_model_1_4.pth'))
        self.model.eval()
        self.AI_move()

    def _make_move(self, row, col):
        """Обрабатывает ход игрока."""
        if self.game.board[row][col] != 0:
            return  # Ячейка уже занята

        # Ход игрока
        self.game.move(2, (row, col))
        self.buttons[row][col]["text"] = "O"

        # Проверка на победу
        if self.game.check_win(2):
            messagebox.showinfo("Победа!", "Игрок (O) выиграл!")
            self._reset_game()
            return

        # Проверка на ничью
        if self.game.is_full():
            messagebox.showinfo("Ничья", "Игра закончилась вничью!")
            self._reset_game()
            return

        # Запускаем ход ИИ с задержкой, чтобы обновить интерфейс
        self.window.after(500, self.AI_move)

    def AI_move(self):
        """Ход ИИ."""
        with torch.no_grad():
            # Преобразуем игровое поле в тензор
            state_tensor = torch.tensor(self.game.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            state_tensor = state_tensor / 2.0
            q_values = self.model(state_tensor)[0].cpu()  # Q-значения для текущего состояния

            # Создаем маску доступных ходов
            available_moves = [(row, col) for row in range(5) for col in range(5) if self.game.board[row][col] == 0]
            mask = torch.tensor([1 if self.game.board[row][col] == 0 else 0 for row in range(5) for col in range(5)],
                                dtype=torch.bool)

            # Применяем маскирование
            masked_q_values = torch.where(mask, q_values, torch.tensor(float('-inf'), dtype=q_values.dtype))
            action = masked_q_values.argmax().item()
            row, col = divmod(action, 5)

            # Совершаем ход
            self.game.move(1, (row, col))
            self.buttons[row][col]["text"] = "X"

            # Проверка на победу
            if self.game.check_win(1):
                messagebox.showinfo("Победа!", "ИИ (X) выиграл!")
                self._reset_game()
                return

            # Проверка на ничью
            if self.game.is_full():
                messagebox.showinfo("Ничья", "Игра закончилась вничью!")
                self._reset_game()
                return

    def _reset_game(self):
        super()._reset_game()
        self.AI_move()

class GameWithAI_O(GameUI):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = QModel().to(self.device)  # Загружаем модель на CUDA/CPU
        self.model.load_state_dict(torch.load('D:/Programs/Tima/NN/models/second/o/q_model_2_4.pth'))
        self.model.eval()

    def _make_move(self, row, col):
        """Обрабатывает ход игрока."""
        if self.game.board[row][col] != 0:
            return  # Ячейка уже занята

        # Ход игрока
        self.game.move(1, (row, col))
        self.buttons[row][col]["text"] = "X"

        # Проверка на победу
        if self.game.check_win(1):
            messagebox.showinfo("Победа!", "Игрок (X) выиграл!")
            self._reset_game()
            return

        # Проверка на ничью
        if self.game.is_full():
            messagebox.showinfo("Ничья", "Игра закончилась вничью!")
            self._reset_game()
            return

        # Запускаем ход ИИ с задержкой, чтобы обновить интерфейс
        self.window.after(500, self.AI_move)

    def AI_move(self):
        """Ход ИИ."""
        with torch.no_grad():
            # Преобразуем игровое поле в тензор
            state_tensor = torch.tensor(self.game.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            state_tensor = state_tensor / 2.0
            q_values = self.model(state_tensor)[0].cpu()  # Q-значения для текущего состояния

            # Создаем маску доступных ходов
            available_moves = [(row, col) for row in range(5) for col in range(5) if self.game.board[row][col] == 0]
            mask = torch.tensor([1 if self.game.board[row][col] == 0 else 0 for row in range(5) for col in range(5)],
                                dtype=torch.bool)

            # Применяем маскирование
            masked_q_values = torch.where(mask, q_values, torch.tensor(float('-inf'), dtype=q_values.dtype))
            action = masked_q_values.argmax().item()
            row, col = divmod(action, 5)

            # Совершаем ход
            self.game.move(2, (row, col))
            self.buttons[row][col]["text"] = "O"

            # Проверка на победу
            if self.game.check_win(2):
                messagebox.showinfo("Победа!", "ИИ (O) выиграл!")
                self._reset_game()
                return

            # Проверка на ничью
            if self.game.is_full():
                messagebox.showinfo("Ничья", "Игра закончилась вничью!")
                self._reset_game()
                return

def play_X():
    app = GameWithAI_X()
    app.run()

def play_O():
    app = GameWithAI_O()
    app.run()


