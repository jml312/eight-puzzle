from enum import Enum
from random import choice, seed


class DIRECTION(str, Enum):
    '''Possible moves of the blank cell.'''

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class HEURISTIC(str, Enum):
    '''Possible heuristics to use for the search.'''

    H1 = "h1"
    H2 = "h2"


class State:
    '''The state of an n x n puzzle. Contains the current state of the puzzle, the position of the blank cell, and the goal state.'''

    BLANK_VALUE = 0
    BLANK_VALUE_STR = "b"
    STATE_NOT_SET = "State is not set."

    def __init__(self, rows, columns, goal, state=None):
        self.rows = rows
        self.columns = columns
        self.goal = goal

        if state:
            self.set_state(state)

    def set_state(self, state):
        '''Sets the state of the puzzle.'''

        self.state = state
        self.blank_position = self.find_blank_position()

    def find_blank_position(self):
        '''Finds the position of the blank cell in the state.'''

        assert self.state, self.STATE_NOT_SET

        for i in range(self.rows):
            for j in range(self.columns):
                if self.state[i][j] == self.BLANK_VALUE:
                    return (i, j)

    def print_state(self):
        '''Prints the state of the puzzle.'''

        assert self.state, self.STATE_NOT_SET

        print(self.__repr__())

    def pretty_print_state(self):
        '''Prints the state of the puzzle in a more readable format.'''

        assert self.state, self.STATE_NOT_SET

        max_val_len = len(str(max(max(self.state))))
        col_width = (max_val_len * 3) * self.columns + self.columns + 1

        # convert state to equal length strings
        state_str = []
        for row in self.state:
            curr_row = []
            for cell in row:
                if cell == self.BLANK_VALUE:
                    cell = " "
                if len(str(cell)) < max_val_len:
                    curr_row.append(str(cell) + " " *
                                    (max_val_len - len(str(cell))))
                else:
                    curr_row.append(str(cell))
            state_str.append(curr_row)

        for idx, row in enumerate(state_str):
            is_last_row = idx == self.rows - 1
            print("-" * col_width)
            row_str = ("|" + ((' ' * max_val_len) +
                              "{}" + (' ' * max_val_len) + "|") * self.columns).format(*row)
            print(row_str)
            if is_last_row:
                print("-" * col_width)

    def move(self, direction):
        '''Moves the blank cell in the given direction. Returns a new State object if the move is valid, otherwise returns None.'''

        assert self.state, "State is not set."

        blank_row, blank_col = self.blank_position

        def move_in_direction(new_row, new_col):
            self.state[blank_row][blank_col] = self.state[new_row][new_col]
            self.state[new_row][new_col] = self.BLANK_VALUE
            self.blank_position = (new_row, new_col)
            return self

        if direction == DIRECTION.UP:
            if blank_row == 0:
                return None
            return move_in_direction(blank_row - 1, blank_col)

        elif direction == DIRECTION.DOWN:
            if blank_row == self.rows - 1:
                return None
            return move_in_direction(blank_row + 1, blank_col)

        elif direction == DIRECTION.LEFT:
            if blank_col == 0:
                return None
            return move_in_direction(blank_row, blank_col - 1)

        elif direction == DIRECTION.RIGHT:
            if blank_col == self.columns - 1:
                return None
            return move_in_direction(blank_row, blank_col + 1)

    def randomize_state(self, n):
        '''Randomizes the state of the puzzle by making n random moves from the goal state.'''

        self.set_state([row[:] for row in self.goal])
        seed(0)
        for _ in range(n):
            self.move(choice(list(DIRECTION)))

    def copy(self):
        '''Returns a copy of the state.'''

        assert self.state, self.STATE_NOT_SET

        return State(rows=self.rows, columns=self.columns, goal=self.goal, state=[row[:] for row in self.state])

    def h_score(self, heuristic):
        '''Returns the heuristic score of the state based on the given heuristic.'''

        assert self.state, self.STATE_NOT_SET

        if heuristic == HEURISTIC.H1:
            return self.h1()
        elif heuristic == HEURISTIC.H2:
            return self.h2()

    def h1(self):
        '''Misplaced tiles heuristic.'''

        assert self.state, self.STATE_NOT_SET

        misplaced_tiles = 0
        for i in range(self.rows):
            for j in range(self.columns):
                if self.state[i][j] != self.BLANK_VALUE and self.state[i][j] != self.goal[i][j]:
                    misplaced_tiles += 1
        return misplaced_tiles

    def h2(self):
        '''Manhattan distance heuristic.'''

        assert self.state, self.STATE_NOT_SET

        def find_target_position(cell):
            for i in range(self.rows):
                for j in range(self.columns):
                    if self.goal[i][j] == cell:
                        return (i, j)

        manhattan_distance = 0
        for i in range(self.rows):
            for j in range(self.columns):
                if self.state[i][j] != self.BLANK_VALUE:
                    curr_row, curr_col = (i, j)
                    target_row, target_col = find_target_position(
                        self.state[i][j])
                    manhattan_distance += abs(curr_row - target_row) + abs(
                        curr_col - target_col)
        return manhattan_distance

    def __repr__(self):
        '''Returns a string representation of the state.'''

        assert self.state, self.STATE_NOT_SET

        return " ".join(["".join(map(
            lambda x: str(x) if x != self.BLANK_VALUE else "b", cell)) for cell in self.state])
