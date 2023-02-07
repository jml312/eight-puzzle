from state import DIRECTION


class StateNode:
    def __init__(self, state, g_score, h_score, heuristic, direction=None, parent=None):
        self.state = state
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = self.g_score + self.h_score
        self.heuristic = heuristic
        self.direction = direction
        self.parent = parent

    def find_children(self):
        children = []
        for direction in DIRECTION:
            child_state = self.state.copy().move(direction)
            if child_state:
                child_node = StateNode(
                    state=child_state, g_score=self.g_score + 1, h_score=child_state.h_score(self.heuristic), heuristic=self.heuristic, direction=direction, parent=self)
                children.append(child_node)
        return children

    def __repr__(self):
        return self.state.__repr__()

    def __lt__(self, other):
        return self.f_score < other.f_score
