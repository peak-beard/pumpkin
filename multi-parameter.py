import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import random




style.use('ggplot')
df = pd.read_csv('pumpkin_data.csv')
plt.scatter(df['pumpkin_diameter'], df['rubber-band_number'])
plt.scatter(df['pumpkin_thickness'], df['rubber-band_number'])

m = len(df['pumpkin_diameter'])



#print(X.shape)
y = df['rubber-band_number']
#X = np.ones((m, 2))
#for i in range(m):
#    X[i, 1] = data[1, i]
X = df['pumpkin_diameter']
X1 = df['pumpkin_thickness']

theta = random.random()
theta1 = random.random()
theta2 = random.random()
def h(X, X1, theta, theta1, theta2) :
    prediction = (X * theta1) + (X1 * theta2) + theta
    return prediction

def err_sum(X, X1, y, theta, theta1, theta2):
    m = len(y)
    err_sum = 0
    prediction = h(X, X1, theta, theta1, theta2)
    for i in range(m):
        temp = prediction[i] - y[i]
        err_sum = temp
    return err_sum

def err_sum1(X, X1, y, theta, theta1, theta2):
    m = len(y)
    err_sum = 0
    prediction = h(X, X1, theta, theta1, theta2)
    for i in range(m):
        temp = (prediction[i] - y[i]) * X[i]
        err_sum = temp
    return err_sum

def err_sum2(X, X1, y, theta, theta1, theta2):
    m = len(y)
    err_sum = 0
    prediction = h(X, X1, theta, theta1, theta2)
    for i in range(m):
        temp = (prediction[i] - y[i]) * X1[i]
        err_sum = temp
    return err_sum
def sqr_err(X, X1, y, theta, theta1, theta2):
    m = len(y)
    sqr_err = 0
    prediction = h(X, X1, theta, theta1, theta2)
    for i in range(m):
        temp = np.square(prediction[i] - y[i]) * (1/m)
        sqr_err = temp
    return sqr_err

def gradient_descent(X, X1, y, theta, theta1, theta2, learning_rate=0.01, iterations=100):
    m = len(y)
    for i in range(iterations):
        temp0 = theta - learning_rate * (1/m) * err_sum(X, X1, y, theta, theta1, theta2)
        temp1 = theta1 - learning_rate * (1/m) * err_sum1(X, X1, y, theta, theta1, theta2)
        temp2 = theta2 - learning_rate * (1 / m) * err_sum2(X, X1, y, theta, theta1, theta2)
        theta = temp0
        theta1 = temp1
        theta2 = temp2
    return theta, theta1, theta2


theta, theta1, theta2 = gradient_descent(X, X1, y, theta, theta1, theta2)
plt.plot(h(X, X1, theta, theta1, theta2))
msg = 'r^2=' + str(sqr_err(X, X1, y, theta, theta1, theta2))
msg2 = 'y = ' + str(theta1) + 'x' + ' + ' + str(theta2) + 'x[1]' + ' + ' + str(theta)
plt.figtext(.1, .9, msg)
plt.figtext(.1, .85, msg2)
plt.show()
print(h(X, X1, theta, theta1, theta2))
