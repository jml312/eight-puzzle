from enum import Enum
from random import seed, choice


class CELL(str, Enum):
    BLANK = "b"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"


class DIRECTION(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class HEURISTIC(str, Enum):
    H1 = "h1"
    H2 = "h2"


class State:
    ROWS = 3
    COLUMNS = 3
    GOAL = [[CELL.BLANK.value, CELL.ONE.value, CELL.TWO.value],
            [CELL.THREE.value, CELL.FOUR.value, CELL.FIVE.value],
            [CELL.SIX.value, CELL.SEVEN.value, CELL.EIGHT.value]]

    def __init__(self, state=None):
        seed(1)
        if state:
            self.set_state(state)

    def set_state(self, state):
        self.state = state
        self.blank_position = self.find_blank_position()

    def find_blank_position(self):
        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                if self.state[i][j] == CELL.BLANK.value:
                    return (i, j)

    # def print_state(self):
    #     print(" ".join(["".join(map(str, cell))
    #           for cell in self.state]) + "\n")

    def print_state(self):
        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                print(self.state[i][j], end=' ')
            print()
        print()

    def move(self, direction):
        blank_row, blank_col = self.blank_position

        def move_in_direction(new_row, new_col):
            self.state[blank_row][blank_col] = self.state[new_row][new_col]
            self.state[new_row][new_col] = CELL.BLANK.value
            self.blank_position = (new_row, new_col)
            return State(self.state)

        if direction == DIRECTION.UP:
            if blank_row == 0:
                return None
            return move_in_direction(blank_row - 1, blank_col)
        elif direction == DIRECTION.DOWN:
            if blank_row == self.ROWS - 1:
                return None
            return move_in_direction(blank_row + 1, blank_col)
        elif direction == DIRECTION.LEFT:
            if blank_col == 0:
                return None
            return move_in_direction(blank_row, blank_col - 1)
        elif direction == DIRECTION.RIGHT:
            if blank_col == self.COLUMNS - 1:
                return None
            return move_in_direction(blank_row, blank_col + 1)

    def randomize_state(self, n):
        self.set_state([row[:] for row in self.GOAL])
        for _ in range(n):
            random_direction = choice(list(DIRECTION)).value
            self.move(random_direction)

    def copy(self):
        return State([row[:] for row in self.state])

    def h_score(self, heuristic):
        if heuristic == HEURISTIC.H1:
            return self.h1()
        elif heuristic == HEURISTIC.H2:
            return self.h2()

    def h1(self):
        '''misplaced tiles heuristic'''
        misplaced_tiles = 0
        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                if self.state[i][j] != CELL.BLANK.value and self.state[i][j] != self.GOAL[i][j]:
                    misplaced_tiles += 1
        return misplaced_tiles

    def h2(self):
        '''manhattan distance heuristic'''
        def find_target_position(cell):
            for i in range(self.ROWS):
                for j in range(self.COLUMNS):
                    if self.GOAL[i][j] == cell:
                        return (i, j)

        manhattan_distance = 0
        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                if self.state[i][j] != CELL.BLANK.value:
                    curr_pos = (i, j)
                    target_pos = find_target_position(self.state[i][j])
                    manhattan_distance += abs(curr_pos[0] - target_pos[0]) + abs(
                        curr_pos[1] - target_pos[1])
        return manhattan_distance

    def __repr__(self):
        return "\n".join([" ".join(map(str, cell)) for cell in self.state]) + "\n"
