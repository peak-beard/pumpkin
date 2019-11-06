import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import random

style.use('ggplot')
df = pd.read_csv('pumpkin_data.csv')
plt.scatter(df['pumpkin_diameter'], df['rubber-band_number'])
#plt.scatter(df['pumpkin_thickness'], df['rubber-band_number'])
m = len(df['pumpkin_diameter'])

#print(X.shape)
y = df['rubber-band_number']
#X = np.ones((m, 2))
#for i in range(m):
#    X[i, 1] = data[1, i]
X = df['pumpkin_diameter']

theta = random.random()
theta1 = random.random()
def h(X, theta, theta1) :
    prediction = X * theta1 + theta
    return prediction

def err_sum(X,y,theta,theta1):
    m = len(y)
    err_sum = 0
    prediction = h(X, theta, theta1)
    for i in range(m):
        temp = prediction[i] - y[i]
        err_sum = temp
    return err_sum

def err_sum1(X,y,theta,theta1):
    m = len(y)
    err_sum = 0
    prediction = h(X, theta, theta1)
    for i in range(m):
        temp = (prediction[i] - y[i]) * X[i]
        err_sum = temp
    return err_sum
def sqr_err(X, y, theta, theta1):
    m = len(y)
    sqr_err = 0
    prediction = h(X, theta, theta1)
    for i in range(m):
        temp = np.square(prediction[i] - y[i]) * (1/m)
        sqr_err = temp
    return sqr_err

def gradient_descent(X, y, theta, theta1, learning_rate=0.01, iterations=100):
    m = len(y)
    for i in range(iterations):
        temp0 = theta - learning_rate * (1/m) * err_sum(X, y, theta, theta1)
        temp1 = theta1 - learning_rate * (1/m) * err_sum1(X, y, theta, theta1)
        theta = temp0
        theta1 = temp1
    return theta, theta1


theta, theta1 = gradient_descent(X, y,theta, theta1)
plt.plot(df['pumpkin_diameter'], h(X, theta, theta1))
msg = 'r^2=' + str(sqr_err(X, y, theta, theta1))
msg2 = 'y = ' + str(theta1) + 'x' + ' + ' + str(theta)
plt.figtext(.4, .9, msg)
plt.figtext(.4, .8, msg2)
plt.show()

