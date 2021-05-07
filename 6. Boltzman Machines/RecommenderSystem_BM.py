# Importing the Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # Neural Network module of PyTorch
import torch.nn.parallel  # For Parallel Computations
import torch.utils.data  # tools
from torch.autograd import Variable  # For Stochastic Gradient Descent

# Importing the Dataset

movies = pd.read_csv('ml-1m/movies.dat', sep="::", header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep="::", header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep="::", header=None, engine='python', encoding='latin-1')
'''
sep stands for separator, that is used to separate the files
header=None specifies that there are no headers in our dataset
'''
# Preparing the Training and Test set
training_set = pd.read_csv('ml-100k/u1.base', sep='\t', delimiter='\t')
training_set = np.array(training_set, dtype='int')
# Since our Dataset contains all the integers therefore we are converting it into int type array

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t', sep='\t')
test_set = np.array(test_set, dtype='int')

# Getting the Number of Users and Movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

'''
We've stored total number of movies and users in a variable so as to make 
a MATRIX where rows are no of users, COLUMNS are movies and the cells are the ratings 
'''


# Converting the data into an array with rows as users and columns as movies
def convert(data):
    """
    We are defining this function to convert the training set and the test set
    We here are creating a list of lists instead of numpy 2D arrays
    because pytorch by default takes list as input not 2D arrays
    --> Here we are creating a list containing the ratings for users
    --> Columns contain the movies.

    ---> There will be 943 lists corresponding to the NO. OF USERS
    ----> There will be 1682 columns corresponding to the No. Of MOVIES.
    """
    new_data = []  # Initializing a list
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings  # Making the indexes of the ratings and the id_movies same
        new_data.append(list(ratings))

    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into PyTorch Tensors

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the Ratings into Binary Ratings 0 -> Dislike and 1 -> Like
training_set[
    training_set == 0] = -1  # Replacing all the 0 values to -1 as, everything will now be converted into 0 and 1

training_set[training_set == 1] = 0  # If Rating is 1 or 2 then probably the user didn't like the Movie, therefore 0
training_set[training_set == 2] = 0

training_set[training_set >= 3] = 1  # If Rating is > 2 that means the user liked the movie, therefore 1

test_set[test_set == 1] = 0  # If Rating is 1 or 2 then probably the user didn't like the Movie, therefore 0
test_set[test_set == 2] = 0

test_set[test_set >= 3] = 1  # If Rating is > 2 that means the user liked the movie, therefore 1

# Creating the Architecture of Neural Network
"""
Here we are going to create a class which will containg the number of 
1. Hidden NODES
2. Weights for probability of hidden nodes given the visible nodes
3. Bias for probability of hidden nodes given the visible nodes
4. Bias for visible nodes given the hidden nodes

Functions inside the Class will be
1. Sample the hidden nodes given the visible nodes 
2. Sample the visible nodes given the hidden nodes
"""


class RBM():
    def __init__(self, visible_nodes, hidden_nodes):
        self.W = torch.randn(hidden_nodes, visible_nodes)
        self.a = torch.randn(1, hidden_nodes)  # Bias for Hidden Nodes
        self.b = torch.randn(1, visible_nodes)  # bias for Visible Nodes

    def sample_hidden(self):
