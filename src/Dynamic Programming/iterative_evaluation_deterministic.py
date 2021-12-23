from gridworld import standard_grid
import numpy as np


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
        print("_______________")
        print(*row, sep=" | ")
    print("_______________")


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
        print("_______________")
        print(*row, sep=" | ")
    print("_______________")


def transition_fill(grid, ACTION_SPACE):
    transition_probs = {}
    rewards = {}
    for i in range(grid.rows):
        for j in range(grid.columns):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]
    return transition_probs, rewards

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
    grid = standard_grid()
    ACTION_SPACE = ('U', 'D', 'L', 'R')
    V = {s: 0.0 for s in grid.all_states()}
    # print_policy(policy, grid)
    transition_probs, rewards = transition_fill(grid, ACTION_SPACE)
