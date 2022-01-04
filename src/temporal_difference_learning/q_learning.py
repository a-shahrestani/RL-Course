from matplotlib import pyplot as plt

from src.dynamic_programming.gridworld import standard_grid, ACTION_SPACE, standard_windy_grid_penalized, negative_grid
import numpy as np
from monte_carlo.monte_carlo_eg import max_dict

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


def epsilon_greedy(Q, s, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    # what happens if you don't do this? i.e. eps=0
    p = np.random.random()
    if p < (1 - eps):
        return max_dict(Q[s])[0]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


if __name__ == '__main__':
    # grid = standard_grid()
    grid = negative_grid()
    print_values(grid.rewards, grid)

    Q = {}
    states = grid.all_states()
    state_sample_count = {}
    sample_counts = {}
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
    update_counts = {}

    reward_per_episode = []
    # repeat until convergence
    n_episodes = 10000
    for _ in range(n_episodes):
        s = grid.reset()
        episode_reward = 0
        while not grid.is_terminal(s):
            a = epsilon_greedy(Q, s)
            r = grid.move(a)
            episode_reward += r

            s2 = grid.current_state()
            maxQ = max_dict(Q[s2])[1]

            q_old = Q[s][a]
            Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * maxQ - Q[s][a])

            update_counts[s] = update_counts.get(s, 0) + 1
            s = s2

        reward_per_episode.append(episode_reward)

    plt.title('reward per episode')
    plt.plot(reward_per_episode)
    plt.show()

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)