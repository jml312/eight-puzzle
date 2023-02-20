from state import DIRECTION


class StateNode:
    '''A node in the search tree. Contains a State, the cost to reach the node (g_score), the heuristic cost to reach the goal (h_score), the f score (g_score + h_score), the heuristic used, the move direction to reach the node from its parent, and a reference to its parent node.'''

    def __init__(self, state, heuristic, g_score, direction=None, parent=None):
        self.state = state
        self.heuristic = heuristic
        self.g_score = g_score
        self.h_score = self.state.h_score(self.heuristic)
        self.f_score = self.g_score + self.h_score
        self.direction = direction
        self.parent = parent

    def get_children(self):
        '''Returns a list of StateNodes that are children of the current node (valid moves).'''

        child_nodes = []
        for direction in DIRECTION:
            child_state = self.state.copy().move(direction)
            if child_state:
                child_node = StateNode(state=child_state,
                                       heuristic=self.heuristic,
                                       g_score=self.g_score + 1,
                                       direction=direction,
                                       parent=self)
                child_nodes.append(child_node)
        return child_nodes

    def is_goal(self):
        '''Returns true if the state of the current node is the goal state (h_score == 0).'''

        return self.h_score == 0

    def __lt__(self, other):
        '''Returns true if the f score of the current node is less than the f score of the other node.'''

        return self.f_score < other.f_score

    def __repr__(self):
        '''Returns a string representation of the node.'''

        return self.state.__repr__()
