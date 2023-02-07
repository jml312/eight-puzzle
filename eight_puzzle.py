# TODO: add comments
# TODO: add assertions
# TODO: add tests

from enum import Enum
from sys import argv
from queue import PriorityQueue
from state import State, HEURISTIC
from state_node import StateNode


class ACTION(str, Enum):
    SET_STATE = "setState"
    PRINT_STATE = "printState"
    MOVE = "move"
    RANDOMIZE_STATE = "randomizeState"
    A_STAR = "A-star"
    BEAM = "beam"
    MAX_NODES = "maxNodes"


class InvalidActionError(Exception):
    pass


class EightPuzzle:

    def __init__(self):
        self.state = State()
        self.max_nodes = None

    def process_input(self, input_file_path):
        try:
            with open(input_file_path, "r") as f:
                for line in f.readlines():
                    if not line.isspace():
                        action, *args = line.split()
                        if action == ACTION.SET_STATE:
                            new_state = [list(row) for row in args]
                            self.state.set_state(new_state)

                        elif action == ACTION.PRINT_STATE:
                            self.state.print_state()

                        elif action == ACTION.MOVE:
                            direction = args[0]
                            self.state.move(direction)

                        elif action == ACTION.RANDOMIZE_STATE:
                            n = int(args[0])
                            self.state.randomize_state(n)

                        elif action == "solve":
                            if args[0] == ACTION.A_STAR:
                                heuristic = args[1]
                                self.solve_a_star(heuristic)

                            elif args[0] == ACTION.BEAM:
                                k = int(args[1])
                                self.solve_beam(k)

                        elif action == ACTION.MAX_NODES:
                            n = int(args[0])
                            self.max_nodes = n

                        else:
                            raise InvalidActionError(
                                "Invalid action. Please check your input file.")

        except FileNotFoundError:
            raise FileNotFoundError(
                "Input file not found. Please check your input file path.")

    def print_solution(self, node):
        if not node:
            print("No solution found.")
            return
        solution = []
        while node:
            if node.direction:
                solution.insert(0, node.direction.value)
            node = node.parent
        print("Moves:", len(solution))
        print("Order:", end=" " if solution else None)
        for idx, move in enumerate(solution):
            print(move, end=" -> " if idx != len(solution) - 1 else None)

    def solve_a_star(self, heuristic):
        start_node = StateNode(state=self.state, g_score=0, h_score=self.state.h_score(
            heuristic), heuristic=heuristic, direction=None, parent=None)
        frontier = PriorityQueue()
        frontier.put((start_node.f_score, start_node))
        reached = {}
        visited_count = 0
        generated_count = 0

        while not frontier.empty():
            _, curr_node = frontier.get()

            visited_count += 1
            if self.max_nodes and visited_count > self.max_nodes:
                print(f"Max nodes reached ({self.max_nodes})")
                return None

            if curr_node.h_score == 0:
                # print("Visited:", visited_count)
                # print("Generated:", generated_count)

                self.print_solution(curr_node)
                return curr_node

            for child_node in curr_node.find_children():
                generated_count += 1
                child_node_key = str(child_node)
                if child_node_key not in reached or child_node.f_score < reached[child_node_key].f_score:
                    reached[child_node_key] = child_node
                    frontier.put((child_node.f_score, child_node))

        self.print_solution(None)
        return None

    def solve_beam(self, k):
        pass


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python eight_puzzle.py <input_file_path>")
        exit()
    eight_puzzle = EightPuzzle()
    eight_puzzle.process_input(argv[1])
