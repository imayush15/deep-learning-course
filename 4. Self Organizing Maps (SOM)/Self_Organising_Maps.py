# Importing The Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)

# Training SOM
from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
'''
------> (x,y) is the grid size of the SOM, for example x=10, y=10 implies the grid size is of (10 X 10)
------> input_len is the number of parameters our dataset has, in this case we have x as the input and it has 15 columns
        of input therefore input_len = 15
------> Sigma is the radius of the neighbourhood in the grid
------> learning_rate is the rate at which the weight are updated
        higher the learning rate, faster the convergence
'''
# Initializing the input Weights
som.random_weights_init(x)

# Training the Model
som.train_random(x, num_iteration=100)

# Visualizing the data
from pylab import bone, pcolor, colorbar, plot, show

bone()  # Initializes the window containing the figure
pcolor(som.distance_map().T)  # Putting the different inter neuron distance on the map
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
# Red circle if the customer didn't get approval & green square if the customer got the approval


for i, x in enumerate(x):
    w = som.winner(x)  # Returns the winning node of the map
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor=None, markersize=10,
         markeredgewidth=2)
    '''----> The above line of code is to place the marker in the center of the square 
       ----> w[0], w[1] implies the o-ordinate 0,1 and then we add 0.5 to both the co-ordinates to place it in center 
       ----> Now to know where to put what marker we are using the 'y' which contains the the information wether the 
             customer got the approval or not 
       ----> if y[i] == 0, put the element at 0th position of the markers array;
             if y[i] ==1, put the element at the 1st position of the markers array;
       ----> Same working of the  colors[y[i]]               
    '''

show()

# Finding the Frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(1, 1)], mappings[(4, 1)]), axis=0)
frauds = sc.inverse_transform(frauds)
print(frauds)
