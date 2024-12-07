from tkinter import Tk, ttk
from .game import Game
import tkinter as tk
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


if __name__ == "__main__":
    app = TicTacToeUI()
    app.run()
