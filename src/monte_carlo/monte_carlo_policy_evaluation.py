from src.dynamic_programming.gridworld import standard_grid, ACTION_SPACE, standard_windy_grid_penalized, negative_grid
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


def play_game(grid, policy, max_steps=20):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.sets_tate(start_states[start_idx])
    s = grid.current_state()
    states = [s]
    rewards = [0]
    steps = 0
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        next_state = grid.current_state()
        states.append(next_state)
        rewards.append(r)
        steps += 1
        if steps > max_steps:
            break
        s = next_state

    return states,rewards

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
    print_values(grid.rewards,grid)

    V = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0
    max_iterations = 100
    for iter in range(max_iterations):
        pass
