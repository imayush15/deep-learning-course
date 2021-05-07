# ----------------------------------------> DATA PREPROCESSING <--------------------------

# ------------------>Importing the Libraries<-----------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---->Importing the Training Set<----
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# ----> Feature Scaling <----
'''
It is recommended to use normalization in RNNs whenever there is a sigmoid activation function used in output layer
Normalization formula is :
                        X_norm = (x - x_min)/(x_max - x_min)
'''
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
# feature_range argument implies that allthe output will be in range (0, 1)
scaled_training_set = sc.fit_transform(training_set)
# print(len(scaled_training_set))
# print(scaled_training_set)

# Creating a data-structure with 60 timesteps and 1 output
'''
---> Here we are creating a data structure for RNN to memorize the previous 60 values and based on that
predict the value at the next day.

---> Model, at particular time 't' will look at the previous 60 values of stock price to 
predict the stock price at time 't+1'.
'''

x_train = []  # This will contain the 60 values at any time 't' for the prediction of value at time 't+1'
y_train = []  # This will contain the output at time "t+1"

for i in range(60, len(scaled_training_set)):
    x_train.append(scaled_training_set[i - 60:i, 0])
    y_train.append(scaled_training_set[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
# Converting x_train and y_train into numpy array to be inputted into RNN model

# Reshaping the data
'''
Since we are going to give x_train as input to keras library, it is very necessary to convert it into a format which is
suitable for input to keras

Input shape should be : (batch_size, timesteps, input_dim)

input_dims is the number of predictors that is the number of things we want to predict
'''

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
'''
By convention in Python '0' corresponds to the ROWS
and '1' corresponds to the COLUMNS
Therefore, x_train.shape[0] returns number of ROWS and x_train.shape[1] returns number of COLUMNS
'''
# print(X_train)

# BUILDING THE RNN

# Importing the Libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the First LSTM layer and Some Dropout Regulaization

# Dropout Regualization is used to avoid Overfitting
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))
"""
---> Units is the number of Neurons which in this case is set equal to 50
---> return_sequence is the option for creating Stacked LSTM and is this case we are building a stacked LSTM, 
     therefore it is set equal to true
---> input_shape is the shape of the input where we are giving only two parameters : timesteps and input_dim as input, 
     because the number of observations is automatically taken into account 
     
---> Dropout only takes 'rate' as input
     It is the rate at which we are dropping the number of neurons in successive layers
"""

# Adding the Second LSTM layer and Some Dropout Regulaization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the Third LSTM layer and Some Dropout Regulaization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the Fourth LSTM layer and Some Dropout Regulaization
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(rate=0.2))

# Adding the Output Layer
regressor.add(Dense(units=1))

# Compiling the CNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN into Training set
regressor.fit(x_train, y_train, epochs=100, batch_size=32)

# Making the Predictions and Visualizing results

# Getting the Real Stock Price
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values
# print(real_stock_price)

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# axis = 0 implies concatenating along horizontal axis
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
# print(inputs)

'''
Now we need to prepare the test set which will be used as input to predict the output 
'''

# Preparing the Test set
x_test = []

for i in range(60, len(inputs)):
    x_test.append(inputs[i - 60:i, 0])  # Populating x_test with the values of inputs
x_test = np.array(x_test)  # Converting it into np array
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshaping it to a tensor with size given

predicted_stock_price = regressor.predict(x_test)  # Applying the predict method to predict the results

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Inversing the scaling to get the correct range  of values

# Visualising the Results
plt.plot(real_stock_price, color='red', label='REAL GOOGLE STOCK PRICE')
plt.plot(predicted_stock_price, color='blue', label='PREDICTED GOOGLE STOCK PRICE')
plt.title("GOOGLE STOCK PRICE PREDICTION")
plt.xlabel("TIME --->")
plt.ylabel("STOCK PRICE --->")
plt.legend()
plt.show()
