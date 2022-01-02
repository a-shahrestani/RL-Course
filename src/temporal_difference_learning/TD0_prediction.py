from matplotlib import pyplot as plt

from src.dynamic_programming.gridworld import standard_grid, ACTION_SPACE, standard_windy_grid_penalized, negative_grid
import numpy as np

GAMMA = 0.9
ALPHA = 0.1
THRESHOLD = 1e-3
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


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


def print_values(v, g):
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


def epsilon_greedy(policy, s, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    # what happens if you don't do this? i.e. eps=0
    p = np.random.random()
    if p < (1 - eps):
        return policy[s]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


if __name__ == '__main__':
    grid = standard_grid()
    # grid = negative_grid()
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    print_values(grid.rewards, grid)
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    # store max change in V(s) per episode
    deltas = []

    # repeat until convergence
    n_episodes = 10000
    for _ in range(n_episodes):
        s = grid.reset()
        delta = 0
        while not grid.is_terminal(s):
            a = epsilon_greedy(policy, s)
            r = grid.move(a)
            s2 = grid.current_state()
            v_old = V[s]
            V[s] = V[s] + ALPHA * (r + GAMMA * V[s2] - V[s])
            delta = max(delta, np.abs(V[s] - v_old))
            s = s2
        deltas.append(delta)

    plt.plot(deltas)
    plt.show()

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)