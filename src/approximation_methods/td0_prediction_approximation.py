from matplotlib import pyplot as plt
from sklearn.kernel_approximation import RBFSampler, Nystroem
from src.dynamic_programming.gridworld import standard_grid, ACTION_SPACE, standard_windy_grid_penalized, negative_grid
import numpy as np

GAMMA = 0.9
ALPHA = 0.01
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


def gather_samples(grid, n_samples=10000):
    samples = []
    for _ in range(n_samples):
        s = grid.reset()
        samples.append(s)
        while not grid.game_over():
            a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            r = grid.move(a)
            s = grid.current_state()
            samples.append(s)
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

    def predict(self, s):
        x = self.featurizer.transform([s])[0]
        return x @ self.w

    def grad(self, s):
        x = self.featurizer.transform([s])[0]
        return x


if __name__ == '__main__':
    # grid = standard_grid()
    grid = negative_grid()
    greedy_policy = {
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

    model = Model(grid)
    mse_per_episode = []

    n_episode = 10000
    for it in range(n_episode):
        if (it + 1) % 100 == 0:
            print(it + 1)

        s = grid.reset()
        Vs = model.predict(s)
        n_steps = 0
        episode_err = 0
        while not grid.game_over():
            a = epsilon_greedy(greedy_policy, s)
            r = grid.move(a)
            s2 = grid.current_state()

            if grid.is_terminal(s2):
                target = r
            else:
                Vs2 = model.predict(s2)
                target = r + GAMMA * Vs2

            g = model.grad(s)
            err = target - Vs
            model.w += (ALPHA * err * g)

            n_steps += 1
            episode_err = err * err

            s = s2
            Vs = Vs2

        mse = episode_err / n_steps
        mse_per_episode.append(mse)

    plt.plot(mse_per_episode)
    plt.title("MSE per Episode")
    plt.show()

    V = {}
    for s in grid.all_states():
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(greedy_policy, grid)
