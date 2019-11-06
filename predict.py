import pumpkin
import random
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

style.use('ggplot')
df = pd.read_csv('pumpkin_data.csv')
plt.scatter(df['pumpkin_diameter'], df['rubber-band_number'])

theta = random.random()
theta1 = random.random()

y = df['rubber-band_number']
X = df['pumpkin_diameter']

theta, theta1 = pumpkin.gradient_descent(X, y,theta, theta1)