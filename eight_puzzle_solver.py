# TODO: beam search (verify solution) - (consider generate random states function in StateNode)
# TODO: (add comments)
# TODO: add tests

from enum import Enum
from sys import argv
from queue import PriorityQueue
from state import State, CELL, DIRECTION, HEURISTIC
from state_node import StateNode
from random import randint


class EightPuzzleSolver(State):
    '''Solver for the eight puzzle game using either the A* or beam search algorithm.'''

    def __init__(self):
        super().__init__()
        self.max_nodes = None

    def solve_a_star(self, heuristic):
        '''Solves the puzzle using the A* algorithm with the specified heuristic. Returns the solution node (or None), number of generated nodes, and number of visited nodes.'''

        print(f"\nSolving with A-star {heuristic}")

        start_node = StateNode(state=self,
                               g_score=0,
                               heuristic=heuristic,
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
                print(
                    f"** No solution found (max nodes of {self.max_nodes} reached) **")
                return None, num_generated, num_visited

            if curr_node.h_score == 0:
                self.print_solution(node=curr_node,
                                    num_generated=num_generated,
                                    num_visited=num_visited)
                return curr_node, num_generated, num_visited

            for child_node in curr_node.get_children():
                num_generated += 1
                child_node_key = str(child_node)
                if child_node_key not in reached or child_node < reached[child_node_key]:
                    reached[child_node_key] = child_node
                    frontier.put((child_node.f_score, child_node))

        print("** No solution found **")
        return None, num_generated, num_visited

    def solve_beam(self, k):
        '''Solves the puzzle using the beam search algorithm with the specified k value. Returns the solution node (or None), number of generated nodes, and number of visited nodes.'''

        print(f"\nSolving with beam {k}")

        start_node = StateNode(state=self,
                               g_score=0,
                               heuristic=HEURISTIC.H2,
                               direction=None,
                               parent=None)
        frontier = PriorityQueue()
        frontier.put((start_node.f_score, start_node))
        reached = {str(start_node): start_node}
        num_generated = 0
        num_visited = 0

        while not frontier.empty():
            for _ in range(len(frontier.queue)):
                _, curr_node = frontier.get()

                num_visited += 1
                if self.max_nodes and num_visited > self.max_nodes:
                    print(
                        f"** No solution found (max nodes of {self.max_nodes} reached) **")
                    return None, num_generated, num_visited

                if curr_node.h_score == 0:
                    self.print_solution(node=curr_node,
                                        num_generated=num_generated,
                                        num_visited=num_visited)
                    return curr_node, num_generated, num_visited

                for child_node in curr_node.get_children():
                    num_generated += 1
                    child_node_key = str(child_node)
                    if child_node_key not in reached or child_node < reached[child_node_key]:
                        reached[child_node_key] = child_node
                        frontier.put((child_node.f_score, child_node))

            frontier.queue = frontier.queue[:k]

        print("** No solution found **")
        return None, num_generated, num_visited

    def print_path(self, solution):
        '''Prints the path from the start state to the goal state.'''

        for idx, node in enumerate(solution):
            is_last_move = idx == len(solution) - 1
            print("Start" if idx ==
                  0 else f"Move {idx}: {node.direction.value}")
            if is_last_move:
                print("** Goal **")
            node.state.pretty_print_state()
            if not is_last_move:
                line = f"{' ' * State.COLUMNS * 2}|{' ' * State.COLUMNS * 2}"
                arrow = f"{' ' * (State.COLUMNS * 2 - 1)}\\'/{' ' * (State.COLUMNS * 2 - 1)}"
                print(f"\n{line}\n{line}\n{arrow}\n")

    def print_solution(self, node=None, num_generated=None, num_visited=None, print_path=False):
        '''Prints the solution to the puzzle.'''

        print("** Solution found **")

        solution = []
        while node:
            solution.insert(0, node)
            node = node.parent

        if print_path:
            self.print_path(solution)
        if num_generated:
            print("Generated:", num_generated)
        if num_visited:
            print("Visited:", num_visited)

        solution = [s for s in solution if s.direction]
        print("Moves:", len(solution))
        print("Order:", end=" ")
        print(" -> ".join([s if isinstance(s, str)
              else s.direction.value for s in ["START"] + solution + ["GOAL"]]))


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
        raise InvalidActionError("\"setState <row1> <row2> <row3>\"")

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
        print("Usage: python eight_puzzle_solver.py <input_file_path>")
        exit()
    try:
        input_file_path = argv[1]
        with open(input_file_path, "r") as f:
            eight_puzzle_solver = EightPuzzleSolver()
            lines = [line for line in f.readlines() if not line.isspace()]

            for line in lines:
                action_args = line.split()

                if line.startswith(ACTION.SET_STATE):
                    valid_arg_length = len(action_args) == 4
                    if not valid_arg_length:
                        action_usage(ACTION.SET_STATE)

                    valid_cells = all(
                        [cell in CELL._value2member_map_ for row in "".join(action_args[1:])for cell in row])
                    valid_duplicates = len(
                        set("".join(action_args[1:]))) == State.ROWS * State.COLUMNS
                    if not valid_cells or not valid_duplicates:
                        action_usage(ACTION.SET_STATE)

                    new_state = [list(row) for row in action_args[1:]]
                    eight_puzzle_solver.set_state(new_state)

                elif line.startswith(ACTION.RANDOMIZE_STATE):
                    valid_arg_length = len(action_args) == 2
                    if not valid_arg_length:
                        action_usage(ACTION.RANDOMIZE_STATE)

                    valid_num_moves = action_args[1].isdigit() and int(
                        action_args[1]) > 0
                    if not valid_num_moves:
                        action_usage(ACTION.RANDOMIZE_STATE)

                    n = int(action_args[1])
                    eight_puzzle_solver.randomize_state(n)

                elif line.startswith(ACTION.PRINT_STATE):
                    valid_arg_length = len(action_args) == 1
                    if not valid_arg_length:
                        action_usage(ACTION.PRINT_STATE)

                    eight_puzzle_solver.pretty_print_state()

                elif line.startswith(ACTION.MOVE):
                    valid_arg_length = len(action_args) == 2
                    if not valid_arg_length:
                        action_usage(ACTION.MOVE)

                    valid_direction = action_args[1] in DIRECTION._value2member_map_
                    if not valid_direction:
                        action_usage(ACTION.MOVE)

                    direction = action_args[1]
                    eight_puzzle_solver.move(direction)

                elif line.startswith(ACTION.MAX_NODES):
                    valid_arg_length = len(action_args) == 2
                    if not valid_arg_length:
                        action_usage(ACTION.MAX_NODES)

                    valid_num_nodes = action_args[1].isdigit() and int(
                        action_args[1]) > 0
                    if not valid_num_nodes:
                        action_usage(ACTION.MAX_NODES)

                    n = int(action_args[1])
                    eight_puzzle_solver.max_nodes = n

                elif line.startswith(ACTION.A_STAR):
                    valid_arg_length = len(action_args) == 3
                    if not valid_arg_length:
                        action_usage(ACTION.A_STAR)

                    valid_heuristic = action_args[2] in HEURISTIC._value2member_map_
                    if not valid_heuristic:
                        action_usage(ACTION.A_STAR)

                    heuristic = action_args[2]
                    eight_puzzle_solver.solve_a_star(heuristic)

                elif line.startswith(ACTION.BEAM):
                    valid_arg_length = len(action_args) == 3
                    if not valid_arg_length:
                        action_usage(ACTION.BEAM)

                    valid_k = action_args[2].isdigit() and int(
                        action_args[2]) > 0
                    if not valid_k:
                        action_usage(ACTION.BEAM)

                    k = int(action_args[2])
                    eight_puzzle_solver.solve_beam(k)

                else:
                    valid_actions = ", ".join(
                        [action.value for action in ACTION])
                    raise InvalidActionError(
                        f"\"{line}\" is not a valid action. Valid actions include: {valid_actions}")

    except FileNotFoundError:
        raise FileNotFoundError(f"File \"{argv[1]}\" not found")
