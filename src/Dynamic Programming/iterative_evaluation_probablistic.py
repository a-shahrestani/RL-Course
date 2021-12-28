from gridworld import standard_windy_grid, ACTION_SPACE
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


def transition_fill(grid, ACTION_SPACE):
    transition_probs = {}
    rewards = {}
    # for i in range(grid.rows):
    #     for j in range(grid.columns):
    #         s = (i, j)
    #         if not grid.is_terminal(s):
    #             for a in ACTION_SPACE:
    #                 if (s,a) in grid.probs.keys():
    #                     for next_possible_state in grid.probs.get((s, a)):
    #                         transition_probs[(s, a, next_possible_state[0])] = next_possible_state[1]
    #                         if next_possible_state in grid.rewards:
    #                             rewards[(s, a, next_possible_state[0])] = grid.rewards.get(next_possible_state[0],0)
    for (s, a), v in grid.probs.items():
        for s2, p in v.items():
            transition_probs[(s, a, s2)] = p
            rewards[(s, a, s2)] = grid.rewards.get(s2, 0)
    return transition_probs, rewards


if __name__ == '__main__':
    policy = {
        (2, 0): {'U': 0.5, 'R': 0.5},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'U': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'U': 1.0},
        (2, 3): {'L': 1.0},
    }
    grid = standard_windy_grid()
    V = {s: 0.0 for s in grid.all_states()}
    # print_policy(policy, grid)
    transition_probs, rewards = transition_fill(grid, ACTION_SPACE)
    print_policy(policy, grid)
    gamma = 0.9
    threshold = 1e-3
    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        action_prob = policy[s ].get(a,0)
                        r = rewards.get((s, a, s2), 0)
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        print("iter: ", it, 'biggest change:', biggest_change)
        print_values(V, grid)
        it += 1
        if biggest_change < threshold:
            break
    print('\n\n')
