from gridworld import Gridworld


def print_policy(p, g):
    rows = g.rows
    columns = g.columns
    grid = []
    for i in range(rows):
        grid_row = []
        for j in range(columns):
            grid_row.append(p.get((i, j), ' '))
        grid.append(grid_row)
    for row in grid:
        print(*row, sep=" | ")


def print_value(v, g):
    rows = g.rows
    columns = g.columns
    grid = []
    for i in range(rows):
        grid_row = []
        for j in range(columns):
            grid_row.append(v.get((i, j), 0.0))
        grid.append(grid_row)
    for row in grid:
        print(*row, sep=" | ")

    def policy_evalution():
        pass


if __name__ == '__main__':
    policy = {(0, 0): 'R',
              (0, 1): 'R',
              (0, 1): 'R',
              (0, 2): 'R',
              (1, 0): 'U',
              (1, 2): 'U',
              (2, 0): 'U',
              (2, 1): 'R',
              (2, 2): 'U',
              (2, 3): 'L',
              }
    grid = Gridworld(3, 4, (2, 0))
    rewards = {(0, 0): 1, (1, 3): -1}
    actions = {(0, 1): ('D', 'R'),
               (0, 2): ('L', 'D', 'R'),
               (1, 0): ('U', 'D'),
               (1, 2): ('U', 'D', 'R'),
               (2, 0): ('U', 'R'),
               (2, 1): ('L', 'R'),
               (2, 2): ('L', 'R', 'U'),
               (2, 3): ('U', 'L'),
               }
    grid.set(actions, rewards)
    values = {(i,j): 0.0 for i in range(grid.rows) for j in range(grid.columns)}
    print_policy(policy, grid)
