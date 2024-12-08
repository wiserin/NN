class Game:
    def __init__(self):
        self.board = [[0 for _ in range(5)] for _ in range(5)]


    def move(self, player, xy) -> None:
        self.board[xy[0]][xy[1]] = player

    def check_win(self, player) -> bool:
        """
        Проверяет, выиграл ли указанный игрок на поле 5x5 с условием ряда из 4 символов.

        :param player: Символ игрока ('X' или 'O')
        :return: True, если игрок выиграл, иначе False
        """
        size = 5  # Размер поля
        win_condition = 4  # Сколько символов нужно для победы

        # Проверка строк
        for row in range(size):
            for col in range(size - win_condition + 1):
                if all(self.board[row][col + i] == player for i in range(win_condition)):
                    return True

        # Проверка столбцов
        for col in range(size):
            for row in range(size - win_condition + 1):
                if all(self.board[row + i][col] == player for i in range(win_condition)):
                    return True

        # Проверка главных диагоналей
        for row in range(size - win_condition + 1):
            for col in range(size - win_condition + 1):
                if all(self.board[row + i][col + i] == player for i in range(win_condition)):
                    return True

        # Проверка побочных диагоналей
        for row in range(size - win_condition + 1):
            for col in range(win_condition - 1, size):
                if all(self.board[row + i][col - i] == player for i in range(win_condition)):
                    return True

        return False

    def is_full(self) -> bool:
        """Проверяет, заполнено ли поле полностью."""
        return all(cell != 0 for row in self.board for cell in row)

    def can_win_on_this_turn(self, player):
        """
        Проверяет, может ли игрок победить на текущем ходу на доске 5x5.

        :param board: 2D-список или NumPy массив (0 — пусто, 1 — крестик, 2 — нолик).
        :param player_symbol: int, символ игрока (1 для крестиков, 2 для ноликов).
        :return: bool, True если игрок может победить на этом ходу, иначе False.
        """
        for row in range(5):
            for col in range(5):
                if self.board[row][col] == 0:  # Если клетка свободна
                    # Сделать временный ход
                    self.board[row][col] = player
                    # Проверить, приводит ли ход к победе
                    if self.check_win(player):
                        self.board[row][col] = 0  # Откатить ход
                        return True
                    self.board[row][col] = 0  # Откатить ход
        return False
    
    def new_board(self):
        self.board = [[0 for _ in range(5)] for _ in range(5)]