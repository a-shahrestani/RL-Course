from matplotlib import pyplot as plt

from src.dynamic_programming.gridworld import standard_grid, ACTION_SPACE, standard_windy_grid_penalized, negative_grid
import numpy as np
import pandas as pd

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
    s = grid.reset()
    a = epsilon_greed(policy,s)

    states = [s]
    rewards = [0]
    actions = [a]
    for _ in range(max_steps):
        r = grid.move(a)
        next_state = grid.current_state()
        states.append(next_state)
        rewards.append(r)
        if grid.game_over():
            break
        else:
            a = epsilon_greed(policy,next_state)
            actions.append(a)

    return states, rewards, actions


def max_dict(d):
    max_val = max(d.values())
    max_keys = [key for key, val in d.items() if val == max_val]
    return np.random.choice(max_keys), max_val


def epsilon_greed(policy, s, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return policy[s]
    else:
        return np.random.choice(ACTION_SPACE)


if __name__ == '__main__':
    grid = standard_grid()
    # grid = negative_grid()

    # randomly initializing the policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)
    print_values(grid.rewards, grid)

    GAMMA = 0.9
    epsilon = 0.2
    Q = {}
    states = grid.all_states()
    state_sample_count = {}
    sample_counts = {}
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            sample_counts[s] = {}
            state_sample_count[s] = 0
            for a in ACTION_SPACE:
                Q[s][a] = 0
                sample_counts[s][a] = 0
        # else:

    delta = []  # keeping track of the changes in returned values
    max_iterations = 10000
    for it in range(max_iterations):
        if it % 1000 == 0:
            print(it)
        biggest_change = 0
        states, rewards, actions = play_game(grid, policy)
        states_actions = list(zip(states, actions))
        T = len(states)
        G = 0
        for t in range(T - 2, -1, -1):
            s = states[t]
            a = actions[t]
            r = rewards[t + 1]
            G = r + GAMMA * G


            if (s, a) not in states_actions[:t]:
                old_q = Q[s][a]
                sample_counts[s][a] += 1
                lr = 1 / sample_counts[s][a]
                Q[s][a] = old_q + lr * (G - old_q)
                state_sample_count[s] +=1
                policy[s] = max_dict(Q[s])[0]
                biggest_change = max(np.abs(Q[s][a] - old_q), biggest_change)
        delta.append(biggest_change)
    plt.plot(delta)
    plt.show()

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("values:")
    print_values(V, grid)
    print("state_sample_count:")
    state_sample_count_arr = np.zeros((grid.rows, grid.columns))
    for i in range(grid.rows):
        for j in range(grid.columns):
            if (i, j) in state_sample_count:
                state_sample_count_arr[i, j] = state_sample_count[(i, j)]
    df = pd.DataFrame(state_sample_count_arr)
    print(df)
