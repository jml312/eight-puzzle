# TODO: clean print_solution + not found + max nodes
# TODO: add comments
# TODO: consider converting solve_beam to use priority queue
# TODO: add assertions
# TODO: add tests

from enum import Enum
from sys import argv
from queue import PriorityQueue
from state import State, HEURISTIC
from state_node import StateNode


class ACTION(str, Enum):
    '''Possible actions to perform on the puzzle.'''

    SET_STATE = "setState"
    PRINT_STATE = "printState"
    MOVE = "move"
    RANDOMIZE_STATE = "randomizeState"
    A_STAR = "A-star"
    BEAM = "beam"
    MAX_NODES = "maxNodes"


class InvalidActionError(Exception):
    '''Raised when an invalid action is encountered in the input file.'''

    pass


class EightPuzzle(State):
    '''The eight puzzle game. Takes an input file as an argument and performs the actions specified in the file on the puzzle.'''

    def __init__(self):
        super().__init__()
        self.max_nodes = None

    def process_input(self, input_file_path):
        '''Processes the input file and performs the actions specified in the file on the puzzle.'''

        try:
            with open(input_file_path, "r") as f:
                lines = [line for line in f.readlines() if not line.isspace()]
                for line in lines:
                    action, *args = line.split()

                    if action == ACTION.SET_STATE:
                        new_state = [list(row) for row in args]
                        self.set_state(new_state)

                    elif action == ACTION.PRINT_STATE:
                        self.pretty_print_state()

                    elif action == ACTION.MOVE:
                        direction = args[0]
                        self.move(direction)

                    elif action == ACTION.RANDOMIZE_STATE:
                        n = int(args[0])
                        self.randomize_state(n)

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

    def print_path(self, solution):
        '''Prints the path from the start state to the goal state.'''

        for idx, node in enumerate(solution):
            is_last_move = idx == len(solution) - 1
            if idx == 0:
                print("Start")
            else:
                print(f"Move {idx}: {node.direction.value}")
                if is_last_move:
                    print("** Goal **")
            node.state.pretty_print_state()

            if not is_last_move:
                print()
                line = " " * State.COLUMNS * 2 + "|" + " " * State.COLUMNS * 2
                print(line + "\n" + line)
                arrow = " " * (State.COLUMNS * 2 - 1) + "\\\'/" + \
                    " " * (State.COLUMNS * 2 - 1) + "\n"
                print(arrow)
            else:
                print()

    def print_solution(self, search_type, search_input, node=None, num_visited=None, num_generated=None, print_path=False):
        print(f"\n** Solution found ({search_type} {search_input}) **\n")

        if node is None:
            if num_visited:
                print("Visited:", num_visited)
            if num_generated:
                print("Generated:", num_generated)
            print("** No solution found **")
            return

        solution = []
        while node:
            solution.insert(0, node)
            node = node.parent

        if print_path:
            self.print_path(solution)

        if num_visited:
            print("Visited:", num_visited)
        if num_generated:
            print("Generated:", num_generated)

        solution = [s for s in solution if s.direction]
        print("Moves:", len(solution))
        print("Order:", end=" " if solution else None)
        for idx, node in enumerate(solution):
            is_last_move = idx == len(solution) - 1
            print(node.direction.value, end=" -> " if not is_last_move else None)

    def solve_a_star(self, heuristic):
        '''Solves the puzzle using the A* algorithm with the specified heuristic. Returns the solution node if a solution is found, otherwise returns None.'''

        start_node = StateNode(state=self,
                               g_score=0,
                               h_score=self.h_score(heuristic),
                               heuristic=heuristic,
                               direction=None,
                               parent=None)
        frontier = PriorityQueue()
        frontier.put((start_node.f_score, start_node))
        reached = {}
        num_visited = 0
        num_generated = 0

        while not frontier.empty():
            _, curr_node = frontier.get()

            num_generated += 1
            if self.max_nodes and num_visited > self.max_nodes:
                print(f"Max nodes of {self.max_nodes} reached")
                return None

            if curr_node.h_score == 0:
                self.print_solution(search_type="A-star",
                                    search_input=heuristic,
                                    node=curr_node,
                                    num_visited=num_visited,
                                    num_generated=num_generated,
                                    print_path=True)
                return curr_node

            for child_node in curr_node.get_children():
                num_visited += 1
                child_node_key = str(child_node)
                if child_node_key not in reached or child_node < reached[child_node_key]:
                    reached[child_node_key] = child_node
                    frontier.put((child_node.f_score, child_node))

        # self.print_solution(num_visited, num_generated)
        print("No solution found")
        return None

    def solve_beam(self, k):
        '''Solves the puzzle using the beam search algorithm with the specified k value. Returns the solution node if a solution is found, otherwise returns None.'''

        start_node = StateNode(state=self,
                               g_score=0,
                               h_score=self.h_score(HEURISTIC.H2),
                               heuristic=HEURISTIC.H2,
                               direction=None,
                               parent=None)
        current_nodes = [start_node]
        num_visited = 0
        num_generated = 0

        while current_nodes:
            candidate_nodes = []

            for curr_node in current_nodes:
                num_generated += 1
                if self.max_nodes and num_visited > self.max_nodes:
                    print(f"Max nodes of {self.max_nodes} reached")
                    return None

                if curr_node.h_score == 0:
                    self.print_solution(search_type="Beam",
                                        search_input=k,
                                        node=curr_node,
                                        num_visited=num_visited,
                                        num_generated=num_generated + 1,
                                        print_path=True)
                    return curr_node

                children_nodes = curr_node.get_children()
                num_visited += len(children_nodes)
                candidate_nodes.extend(children_nodes)

            current_nodes = sorted(
                candidate_nodes, key=lambda x: x.f_score)[:k]

        self.print_solution(num_visited, num_generated)
        return None


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python eight_puzzle.py <input_file_path>")
        exit()
    eight_puzzle = EightPuzzle()
    eight_puzzle.process_input(argv[1])
