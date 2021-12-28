class Gridworld:

    def __init__(self, rows, columns, start):
        self.actions = None
        self.rewards = None
        self.rows = rows
        self.columns = columns
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        self.actions = actions
        self.rewards = rewards

    def sets_tate(self, state):
        self.i = state[0]
        self.j = state[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, state):
        return state not in self.actions

    def get_next_state(self, s, a):
        i, j = s[0], s[1]
        if a in self.actions[s]:
            if a == 'U':
                i -= 1
            elif a == 'D':
                i += 1
            elif a == 'R':
                j += 1
            elif a == 'L':
                j -= 1
        return i, j

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        # return a reward (if any)
        return self.rewards.get((self.i, self.j), 0)

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def undo_move(self, action):
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        assert (self.current_state() in self.all_states())

    def game_over(self):
        return self.current_state() not in self.actions


def standard_grid():
    g = Gridworld(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g


if __name__ == '__main__':
    grid = Gridworld(3, 4, (2, 0))
    rewards = {(0, 0): 1, (1, 3): -1}
    actions = {(0, 0): ('D', 'R'),
               (0, 1): ('D', 'R'),
               (0, 2): ('L', 'D', 'R'),
               (1, 0): ('U', 'D'),
               (1, 2): ('U', 'D', 'R'),
               (2, 0): ('U', 'R'),
               (2, 1): ('L', 'R'),
               (2, 2): ('L', 'R', 'U'),
               (2, 3): ('U', 'L'),
               }
    grid.set(rewards, actions)

