'''
In this case study we are trying to find out the the frauds of the credit cards application using.
---> SOM
---> Artificial Neural Networks
We first implement the SOM moddel to obtain a SOM which will be taken as input by the ANN to predcit the number of fraud
customers in the application list
'''

# Importing The Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

# Training SOM
from minisom import MiniSom
som = MiniSom(x=20, y=20, input_len=15, learning_rate=0.5, sigma=1)

# Initializing the Input Weights
som.random_weights_init(x)

# Training the Model
som.train_random(x, num_iteration=100)

# Visualising the Data
from pylab import bone, show, pcolor, colorbar, plot

bone()  # Initializes the window containing the figure
pcolor(som.distance_map().T)  # Putting the different inter neuron distance on the map
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(x):
    w = som.winner(x)  # Returns the winning node of the map
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor=None, markersize=10,
         markeredgewidth=2)

#show()

# Finding the Frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(6,8)], mappings[(5,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)
