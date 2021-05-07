# Importing the Libraries
import numpy as np
import pandas as pd


# Importing the Dataset
dataset = pd.read_csv('E:\Study_Files\Projects\Machine_Learning\Datasets/Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data
    # Encoding 'Gender' Column using label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

    # Encoding 'Geography coulumn' using OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#####################################################
# Building the ANN
#####################################################

# Importing Tensorflow
import tensorflow as tf

# Initializing the ann
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu',))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu',))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN model
ann.fit(x_train, y_train, batch_size=32, epochs = 200)

# Predicting a single result
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])) > 0.5)

# Predicting the test set results
y_pred = ann.predict(x_test)
y_pred = y_pred > 0.5
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1))

# Confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(f'The Confusion Matrix is : {cm}\n')
print(f'The Accuracy Score is : {ac}\n')