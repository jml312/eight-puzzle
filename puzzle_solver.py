# TODO: testing
# TODO: report

from enum import Enum
from sys import argv
from queue import PriorityQueue
from state import State, DIRECTION, HEURISTIC
from state_node import StateNode


class PuzzleSolver(State):
    '''Solver for an n x n sliding puzzle game using either the A* or beam search algorithm.'''

    def __init__(self, rows, columns, goal, no_print=False, show_path=False):
        super().__init__(rows, columns, goal)
        self.max_nodes = None
        self.no_print = no_print
        self.show_path = show_path

    def solve_a_star(self, heuristic):
        '''Solves the puzzle using the A* algorithm with the specified heuristic. Returns the solution node (or None), number of generated nodes, and number of visited nodes.'''

        assert self.state, State.STATE_NOT_SET

        if not self.no_print:
            print(f"\nSolving with A-star {heuristic}")

        start_node = StateNode(state=self,
                               heuristic=heuristic,
                               g_score=0,
                               direction=None,
                               parent=None)
        frontier = PriorityQueue()
        frontier.put((start_node.f_score, start_node))
        reached = {str(start_node): start_node}
        num_generated = 0
        num_visited = 0

        while not frontier.empty():
            _, curr_node = frontier.get()

            num_visited += 1
            if self.max_nodes and num_visited > self.max_nodes:
                if not self.no_print:
                    print(
                        f"** No solution found (max nodes of {self.max_nodes} reached) **")
                return {
                    "generated": num_generated,
                    "visited": num_visited,
                    "moves": 0,
                    "order": [],
                    "success": False,
                }

            if curr_node.is_goal():
                solution = self.get_solution(end_node=curr_node)
                if not self.no_print:
                    self.print_solution(solution=solution,
                                        num_generated=num_generated,
                                        num_visited=num_visited)
                return {
                    "generated": num_generated,
                    "visited": num_visited,
                    "moves": len(solution) - 1,
                    "order": [s.direction.value for s in solution[1:]],
                    "success": True
                }

            for child_node in curr_node.get_children():
                num_generated += 1
                child_node_key = str(child_node)

                if child_node_key not in reached or child_node < reached[child_node_key]:
                    reached[child_node_key] = child_node
                    frontier.put((child_node.f_score, child_node))

        if not self.no_print:
            print("** No solution found **")
        return {
            "generated": num_generated,
            "visited": num_visited,
            "moves": 0,
            "order": [],
            "success": False
        }

    def solve_beam(self, k):
        '''Solves the puzzle using the beam search algorithm with the specified k value. Uses manhattan distance as heuristic. Returns the solution node (or None), number of generated nodes, and number of visited nodes. '''

        assert self.state, State.STATE_NOT_SET

        if not self.no_print:
            print(f"\nSolving with beam {k}")

        start_node = StateNode(state=self,
                               heuristic=HEURISTIC.H2,
                               g_score=0,
                               direction=None,
                               parent=None)
        frontier = [start_node]
        reached = {str(start_node): start_node}
        num_generated = 0
        num_visited = 0

        while frontier:
            for _ in range(len(frontier)):
                curr_node = frontier.pop(0)

                num_visited += 1
                if self.max_nodes and num_visited > self.max_nodes:
                    if not self.no_print:
                        print(
                            f"** No solution found (max nodes of {self.max_nodes} reached) **")
                    return {
                        "generated": num_generated,
                        "visited": num_visited,
                        "moves": 0,
                        "order": [],
                        "success": False,
                    }

                if curr_node.is_goal():
                    solution = self.get_solution(end_node=curr_node)
                    if not self.no_print:
                        self.print_solution(solution=solution,
                                            num_generated=num_generated,
                                            num_visited=num_visited)
                    return {
                        "generated": num_generated,
                        "visited": num_visited,
                        "moves": len(solution) - 1,
                        "order": [s.direction.value for s in solution[1:]],
                        "success": True
                    }

                for child_node in curr_node.get_children():
                    num_generated += 1
                    child_node_key = str(child_node)
                    if child_node_key not in reached or child_node < reached[child_node_key]:
                        reached[child_node_key] = child_node
                        frontier.append(child_node)

            frontier = sorted(frontier, key=lambda node: node.f_score)[:k]

        if not self.no_print:
            print("** No solution found **")
        return {
            "generated": num_generated,
            "visited": num_visited,
            "moves": 0,
            "order": [],
            "success": False
        }

    def get_solution(self, end_node):
        '''Returns the solution path from the start state to the goal state.'''

        assert self.state, State.STATE_NOT_SET

        solution = []
        while end_node:
            solution.insert(0, end_node)
            end_node = end_node.parent

        return solution

    def print_path(self, solution):
        '''Prints the path from the start state to the goal state.'''

        assert self.state, State.STATE_NOT_SET

        for idx, node in enumerate(solution):
            is_last_move = idx == len(solution) - 1
            print("Start" if idx ==
                  0 else f"Move {idx}: {node.direction.value}")
            if is_last_move:
                print("** Goal **")
            node.state.pretty_print_state()
            if not is_last_move:
                line = f"{' ' * self.columns * 2}|{' ' * self.columns * 2}"
                arrow = f"{' ' * (self.columns * 2 - 1)}\\'/{' ' * (self.columns * 2 - 1)}"
                print(f"\n{line}\n{line}\n{arrow}\n")

    def print_solution(self, solution, num_generated=None, num_visited=None):
        '''Prints the solution to the puzzle.'''

        assert self.state, State.STATE_NOT_SET

        print("** Solution found **")

        if not self.no_print and self.show_path:
            self.print_path(solution)
        if num_generated:
            print("Generated:", num_generated)
        if num_visited:
            print("Visited:", num_visited)

        print("Moves:", len(solution) - 1)
        print("Order:", end=" ")
        print(" -> ".join([s if isinstance(s, str)
              else s.direction.value for s in ["START"] + solution[1:] + ["GOAL"]]))


class ACTION(str, Enum):
    '''Possible actions to perform on the puzzle.'''

    SET_STATE = "setState"
    RANDOMIZE_STATE = "randomizeState"
    PRINT_STATE = "printState"
    MOVE = "move"
    MAX_NODES = "maxNodes"
    A_STAR = "solve A-star"
    BEAM = "solve beam"


class InvalidActionError(Exception):
    '''Raised when an invalid action is encountered in the input file.'''

    pass


def action_usage(action):
    '''Raises an InvalidActionError with the correct usage for the given action.'''

    if action == ACTION.SET_STATE:
        raise InvalidActionError(
            "\"setState <row1> <row2> <row3> (cells consist of one of the following values: 1, 2, 3, 4, 5, 6, 7, 8, b)\"")

    elif action == ACTION.RANDOMIZE_STATE:
        raise InvalidActionError("\"randomizeState <num_moves>\"")

    elif action == ACTION.PRINT_STATE:
        raise InvalidActionError("\"printState\"")

    elif action == ACTION.MOVE:
        raise InvalidActionError("\"move <direction>\"")

    elif action == ACTION.MAX_NODES:
        raise InvalidActionError("\"maxNodes <max_nodes>\"")

    elif action == ACTION.A_STAR:
        raise InvalidActionError("\"solve A-star <heuristic>\"")

    elif action == ACTION.BEAM:
        raise InvalidActionError("\"solve beam <k>\"")


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python puzzle_solver.py <input_file_path>")
        exit()
    try:
        input_file_path = argv[1]
        with open(input_file_path, "r") as f:

            # define puzzle solver values for 3x3 puzzle
            rows = 3
            columns = 3
            goal = [[State.BLANK_VALUE, 1, 2], [3, 4, 5], [6, 7, 8]]
            puzzle_values = [
                str(cell) if cell != State.BLANK_VALUE else "b" for row in goal for cell in row]
            puzzle_solver = PuzzleSolver(
                rows=rows, columns=columns, goal=goal, show_path=True)

            lines = [line for line in f.readlines() if not line.isspace()]

            for line in lines:
                action_args = line.split()

                if line.startswith(ACTION.SET_STATE):
                    valid_arg_length = len(action_args) == 4
                    if not valid_arg_length:
                        action_usage(ACTION.SET_STATE)

                    valid_cells = all(
                        cell in puzzle_values for row in "".join(action_args[1:]) for cell in row
                    )
                    valid_duplicates = len(
                        set("".join(action_args[1:]))) == rows * columns
                    if not valid_cells or not valid_duplicates:
                        action_usage(ACTION.SET_STATE)

                    new_state = [
                        [int(cell) if cell != "b" else State.BLANK_VALUE for cell in row] for row in action_args[1:]]
                    puzzle_solver.set_state(new_state)

                elif line.startswith(ACTION.RANDOMIZE_STATE):
                    valid_arg_length = len(action_args) == 2
                    if not valid_arg_length:
                        action_usage(ACTION.RANDOMIZE_STATE)

                    valid_num_moves = action_args[1].isdigit() and int(
                        action_args[1]) > 0
                    if not valid_num_moves:
                        action_usage(ACTION.RANDOMIZE_STATE)

                    n = int(action_args[1])
                    puzzle_solver.randomize_state(n)

                elif line.startswith(ACTION.PRINT_STATE):
                    valid_arg_length = len(action_args) == 1
                    if not valid_arg_length:
                        action_usage(ACTION.PRINT_STATE)

                    puzzle_solver.pretty_print_state()

                elif line.startswith(ACTION.MOVE):
                    valid_arg_length = len(action_args) == 2
                    if not valid_arg_length:
                        action_usage(ACTION.MOVE)

                    valid_direction = action_args[1] in DIRECTION._value2member_map_
                    if not valid_direction:
                        action_usage(ACTION.MOVE)

                    direction = action_args[1]
                    puzzle_solver.move(direction)

                elif line.startswith(ACTION.MAX_NODES):
                    valid_arg_length = len(action_args) == 2
                    if not valid_arg_length:
                        action_usage(ACTION.MAX_NODES)

                    valid_num_nodes = action_args[1].isdigit() and int(
                        action_args[1]) > 0
                    if not valid_num_nodes:
                        action_usage(ACTION.MAX_NODES)

                    n = int(action_args[1])
                    puzzle_solver.max_nodes = n

                elif line.startswith(ACTION.A_STAR):
                    valid_arg_length = len(action_args) == 3
                    if not valid_arg_length:
                        action_usage(ACTION.A_STAR)

                    valid_heuristic = action_args[2] in HEURISTIC._value2member_map_
                    if not valid_heuristic:
                        action_usage(ACTION.A_STAR)

                    heuristic = action_args[2]
                    puzzle_solver.solve_a_star(heuristic)

                elif line.startswith(ACTION.BEAM):
                    valid_arg_length = len(action_args) == 3
                    if not valid_arg_length:
                        action_usage(ACTION.BEAM)

                    valid_k = action_args[2].isdigit() and int(
                        action_args[2]) > 0
                    if not valid_k:
                        action_usage(ACTION.BEAM)

                    k = int(action_args[2])
                    puzzle_solver.solve_beam(k)

                else:
                    valid_actions = ", ".join(
                        [action.value for action in ACTION])
                    raise InvalidActionError(
                        f"\"{line}\" is not a valid action. Valid actions include: {valid_actions}")

    except FileNotFoundError:
        raise FileNotFoundError(f"File \"{argv[1]}\" not found")
