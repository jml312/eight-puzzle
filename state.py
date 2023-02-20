from enum import Enum
from random import choice, seed


class CELL(str, Enum):
    '''Possible cell values of the state.'''

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
    '''The state of the eight puzzle. Contains the current state of the puzzle, the position of the blank cell, and the goal state.'''

    ROWS = 3
    COLUMNS = 3
    GOAL = [[CELL.BLANK.value, CELL.ONE.value, CELL.TWO.value],
            [CELL.THREE.value, CELL.FOUR.value, CELL.FIVE.value],
            [CELL.SIX.value, CELL.SEVEN.value, CELL.EIGHT.value]]

    def __init__(self, state=None, goal=GOAL):
        if state:
            self.set_state(state)
        self.goal = goal

    def set_state(self, state):
        '''Sets the state of the puzzle.'''

        self.state = state
        self.blank_position = self.find_blank_position()

    def find_blank_position(self):
        '''Finds the position of the blank cell in the state.'''

        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                if self.state[i][j] == CELL.BLANK:
                    return (i, j)

    def print_state(self):
        '''Prints the state of the puzzle.'''

        print(self.__repr__())

    def pretty_print_state(self):
        '''Prints the state of the puzzle in a more readable format.'''

        for idx, row in enumerate(self.state):
            row = [cell if cell != CELL.BLANK else " " for cell in row]
            is_last_row = idx == self.ROWS - 1
            print("-" * 13)
            print("| {} | {} | {} |".format(*row))
            if is_last_row:
                print("-" * 13)

    def move(self, direction):
        '''Moves the blank cell in the given direction. Returns a new State object if the move is valid, otherwise returns None.'''

        blank_row, blank_col = self.blank_position

        def move_in_direction(new_row, new_col):
            self.state[blank_row][blank_col] = self.state[new_row][new_col]
            self.state[new_row][new_col] = CELL.BLANK.value
            self.blank_position = (new_row, new_col)
            return self

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
        '''Randomizes the state of the puzzle by making n random moves from the goal state.'''

        seed(0)
        self.set_state([row[:] for row in self.goal])
        for _ in range(n):
            self.move(choice(list(DIRECTION)))

    def copy(self):
        '''Returns a copy of the state.'''

        return State(state=[row[:] for row in self.state], goal=self.goal)

    def h_score(self, heuristic):
        '''Returns the heuristic score of the state based on the given heuristic.'''

        if heuristic == HEURISTIC.H1:
            return self.h1()
        elif heuristic == HEURISTIC.H2:
            return self.h2()

    def h1(self):
        '''Misplaced tiles heuristic.'''

        misplaced_tiles = 0
        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                if self.state[i][j] != CELL.BLANK and self.state[i][j] != self.goal[i][j]:
                    misplaced_tiles += 1
        return misplaced_tiles

    def h2(self):
        '''Manhattan distance heuristic.'''

        def find_target_position(cell):
            for i in range(self.ROWS):
                for j in range(self.COLUMNS):
                    if self.goal[i][j] == cell:
                        return (i, j)

        manhattan_distance = 0
        for i in range(self.ROWS):
            for j in range(self.COLUMNS):
                if self.state[i][j] != CELL.BLANK:
                    curr_row, curr_col = (i, j)
                    target_row, target_col = find_target_position(
                        self.state[i][j])
                    manhattan_distance += abs(curr_row - target_row) + abs(
                        curr_col - target_col)
        return manhattan_distance

    def __repr__(self):
        '''Returns a string representation of the state.'''

        return " ".join(["".join(map(str, cell)) for cell in self.state])
