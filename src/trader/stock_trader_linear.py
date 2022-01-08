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
    def __init__(self,input_dim,n_actions):
        self.W = np.random.randn(input_dim,n_actions)/np.sqrt(input_dim)
        self.b = np.zeros(input_dim)
        # used for momentum in gradient descent
        self.vW = 0
        self.vb = 0
        self.losses = []
