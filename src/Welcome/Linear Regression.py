import numpy as np
import random
import matplotlib.pyplot as plt


def data_generator():
    number_of_points = 100
    m, b = 1, 0
    lower, upper = -100, 100
    x1 = [random.randrange(start=1, stop=50) for i in range(number_of_points)]
    x2 = [random.randrange(start=1, stop=50) for i in range(number_of_points)]
    y1 = [random.randrange(start=lower, stop=m * x + b) for x in x1]
    y2 = [random.randrange(start=m * x + b, stop=upper) for x in x2]
    plt.plot(np.arange(50), m * np.arange(50) + b)
    plt.scatter(x1, y1, c='blue')
    plt.scatter(x2, y2, c='red')
    plt.show()
    train_x = np.array(x1 + x2)
    train_y = np.array(y1 + y2)
    return train_x, train_y


def coefs(train_x, train_y):
    b0, b1 = 0, 0
    mean_x = np.mean(train_x)
    mean_y = np.mean(train_y)
    SS_xy = np.sum(train_y * train_x) - mean_y * mean_x
    SS_xx = np.sum(train_x * train_x) - mean_x * mean_x
    b1 = SS_xy / SS_xx
    b0 = mean_y - b1 * mean_x
    return b0, b1


def linear_regression(train_x, train_y):
    b0, b1 = coefs(train_x, train_y)
    y_pred = b0 + b1 * train_x
    plt.scatter(train_x, train_y)
    plt.plot(train_x, y_pred)
    plt.show()



train_x, train_y = data_generator()
linear_regression(train_x, train_y)

