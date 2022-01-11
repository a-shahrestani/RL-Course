import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler


# reading the data
def get_data():
    df = pd.read_csv('../data/aapl_msi_sbux.csv')
    return df.values


# create a scaler by playing a random episode and fitting the scaler on the observed states.
# can become more accurate by running the code for multiple episodes
def get_scaler(env):
    states = []
    for _ in range(env.n_steps):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


# creating a directory if it does not exist - utility function
def maybe_make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


class LinearModel:
    def __init__(self, input_dim, n_actions):
        self.W = np.random.randn(input_dim, n_actions) / np.sqrt(input_dim)
        self.b = np.zeros(input_dim)
        # used for momentum in gradient descent
        self.vW = 0
        self.vb = 0
        self.losses = []

    def predict(self, X):
        # X must be of shape N * D and 2D
        assert (len(X.shape) == 2)
        print(self.W)
        print(self.b)
        print(X)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        # X must be of shape N * D and 2D
        assert (len(X.shape) == 2)
        num_values = np.prod(Y.shape)
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # updating the momentum
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)


class MultiStockEnv:
    """
      A 3-stock trading environment.
      State: vector of size 7 (n_stock * 2 + 1)
        - # shares of stock 1 owned
        - # shares of stock 2 owned
        - # shares of stock 3 owned
        - price of stock 1 (using daily close price)
        - price of stock 2
        - price of stock 3
        - cash owned (can be used to purchase more stocks)
      Action: categorical variable with 27 (3^3) possibilities
        - for each stock, you can:
        - 0 = sell
        - 1 = hold
        - 2 = buy
      """

    def __init__(self, data, initial_investment=20000):
        # information about the data
        self.stock_price_history = data
        self.n_step, self.n_stock = data.shape

        # controlling variables
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.initial_investment = initial_investment
        self.cash_in_hand = None
        self.action_space = np.arange(3 ** self.n_stock)
        # creates all the permutations of the actions for each stock
        """
        actions:
            sell = 0
            hold = 1
            buy = 2
        """
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        self.state_dim = 2 * self.n_stock + 1
        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        # check if the action is valid
        assert action in self.action_space

        prev_val = self._get_val()
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]
        self._trade()
        cur_val = self._get_val()
        reward = cur_val - prev_val

        done = (self.cur_step == self.n_step - 1)
        info = {'cur_val': cur_val}

        return self._get_obs(), reward, done, info

    # return observation/state
    def _get_obs(self):
        state = np.empty(self.state_dim)
        state[:self.n_stock] = self.stock_owned
        state[2 * self.n_stock:2 * self.n_stock] = self.stock_price
        state[-1] = self.cash_in_hand
        return state

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        action_vec = self.action_list[action]
        sell_indices = []
        buy_indices = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_indices.append(i)
            elif a == 2:
                buy_indices.append(i)
        # the sell action sells all the stocks of the certain stock
        for i in sell_indices:
            self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
            self.stock_owned[i] = 0

        # the buy action buys 1 of each stock until there is no more cash is left
        can_buy = True
        while can_buy:
            for i in buy_indices:
                if self.cash_in_hand > self.stock_price[i]:
                    self.stock_owned[i] += 1  # buy one share
                    self.cash_in_hand -= self.stock_price[i]
                else:
                    can_buy = False


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)
        # So we only calculated the target for the action we took
        # We need to keep the value for the other targets the same
        target_full = self.model.predict(state)
        # We change the value for the action we took to the new target - the others remain the same
        target_full[0, action] = target

        self.model.sgd(state, target_full)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = env.get_scaler().transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info['cur_val']
