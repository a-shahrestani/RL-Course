from matplotlib import pyplot as plt
from numpy import array
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from src.dynamic_programming.gridworld import standard_grid, ACTION_SPACE, standard_windy_grid_penalized, negative_grid
import numpy as np
from monte_carlo.monte_carlo_eg import max_dict

GAMMA = 0.9
ALPHA = 0.1
THRESHOLD = 1e-3
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ACTION2INT = {a: i for i, a in enumerate(ALL_POSSIBLE_ACTIONS)}
INT2ONEHOT = np.eye(len(ALL_POSSIBLE_ACTIONS))


def one_hot(k):
    return INT2ONEHOT[k]


def merge_state_action(s, a):
    ai = one_hot(ACTION2INT[a])
    return np.concatenate((s, ai))


def gather_samples(grid, n_samples=10000):
    samples = []
    for _ in range(n_samples):
        s = grid.reset()
        # samples.append(s)
        while not grid.game_over():
            a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            sa = merge_state_action(s, a)
            samples.append(sa)
            r = grid.move(a)
            s = grid.current_state()
    return samples


class Model:
    def __init__(self, grid):
        # fit the featurizer to data
        samples = gather_samples(grid)
        # self.featurizer = Nystroem()
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)

        dims = self.featurizer.n_components

        self.w = np.zeros(dims)

    def predict(self, s, a):
        sa = merge_state_action(s, a)
        x = self.featurizer.transform([sa])[0]
        return x @ self.w

    def predict_all_actions(self, s):
        return [self.predict(s, a) for a in ALL_POSSIBLE_ACTIONS]

    def grad(self, s, a):
        sa = merge_state_action(s, a)
        x = self.featurizer.transform([sa])[0]
        return x


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


def epsilon_greedy(model, s, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    # what happens if you don't do this? i.e. eps=0
    p = np.random.random()
    if p < (1 - eps):
        values = model.predict_all_actions(s)
        return ALL_POSSIBLE_ACTIONS[np.argmax(values)]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


if __name__ == '__main__':
    # grid = standard_grid()
    grid = negative_grid()
    print_values(grid.rewards, grid)

    model = Model(grid)
    reward_per_episode = []
    state_visit_count = {}

    n_episode = 20000
    for it in range(n_episode):
        if (it + 1) % 100 == 0:
            print(it + 1)

        s = grid.reset()
        a = epsilon_greedy(model, s)
        Qs = model.predict(s, a)
        n_steps = 0
        episode_err = 0
        episode_reward = 0
        while not grid.game_over():
            a = epsilon_greedy(model, s)
            r = grid.move(a)
            s2 = grid.current_state()
            state_visit_count[s2] = state_visit_count.get(s2, 0) + 1
            if grid.is_terminal(s2):
                target = r
            else:
                values = model.predict_all_actions(s2)
                target = r + GAMMA * np.max(values)

            g = model.grad(s, a)
            err = target - model.predict(s, a)
            model.w += (ALPHA * err * g)

            episode_reward += r
            s = s2
        reward_per_episode.append(episode_reward)

plt.plot(reward_per_episode)
plt.title("Reward per Episode")
plt.show()

greedy_policy = {}
V = {}
for s in grid.all_states():
    if s in grid.actions:
        values = model.predict_all_actions(s)
        V[s] = np.max(values)
        greedy_policy[s] = np.argmax(ALL_POSSIBLE_ACTIONS[np.argmax(values)])
    else:
        V[s] = 0

print("values:")
print_values(V, grid)
print("policy:")
print_policy(greedy_policy, grid)
