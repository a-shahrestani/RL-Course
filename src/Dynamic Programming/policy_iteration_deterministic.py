from gridworld import standard_grid, ACTION_SPACE
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


def policy_evaluation(policy, grid, V, transition_probs, rewards, gamma, threshold):
    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        action_prob = 1 if policy.get(s) == a else 0
                        r = rewards.get((s, a, s2), 0)
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        # print("iter: ", it, 'biggest change:', biggest_change)
        print_values(V, grid)
        it += 1
        if biggest_change < threshold:
            break


if __name__ == '__main__':
    grid = standard_grid()
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)
    V = {s: 0.0 for s in grid.all_states()}
    # print_policy(policy, grid)
    transition_probs, rewards = transition_fill(grid, ACTION_SPACE)
    print_policy(policy, grid)
    gamma = 0.9
    threshold = 1e-3
    while True:
        is_policy_converged = True
        policy_evaluation(policy, grid, V, transition_probs, rewards, gamma, threshold)
        for s in grid.actions.keys():
            old_a = policy[s]
            new_a = None
            best_value = float('-inf')
            for a in ACTION_SPACE:
                v = 0
                for s2 in grid.all_states():
                    r = rewards.get((s,a,s2),0)
                    v += transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])
                if v > best_value:
                    best_value = v
                    new_a = a
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False
        if is_policy_converged:
            break

    print("velues:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy,grid)