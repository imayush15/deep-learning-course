# importing the Libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing
#-------------------->>>>>Training Set

# transforming the Training set to avoid overfitting
train_datagen = ImageDataGenerator(
        rescale=1./255, # Rescale is used for Feature Scaling 1/255 implies that it divides all the pixels by 255
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_set = train_datagen.flow_from_directory(
        'E:/Study_Files/Projects/Machine_Learning/Datasets/dataset/training_set',
        target_size=(64, 64), # the dimensions to which all images found will be resized.
        batch_size=32,
        class_mode='binary')

#------------------>>>>> Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory(
        'E:/Study_Files/Projects/Machine_Learning/Datasets/dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

######################
# Building CNN
######################

# Initializing CNN
cnn = tf.keras.models.Sequential()

# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, input_shape=(64, 64, 3), kernel_size=3, activation='relu'))
# We set the input size earlier to 64, 64; therefore input shape of [64,64]
# 3 refers to tha face that we are working with color images RGB, and not Black and white images

# Adding Pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=[2,2], strides = 2))

# Adding Second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Units ----> Number of Hidden layer

# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid'))
# Since we are working with binary classification
# Therefore we are using 1 neuron as output (Either this 'OR' this)

###############################
# Training the CNN
###############################

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Training the CNN on Training set and Evaluating it on Test set
cnn.fit(x = train_set, validation_data = test_set, epochs=25)
# validation_data argument takes the input on which we want to evaluate the model

# Making a single predicition
import numpy as np
from keras.preprocessing import image

test_img = image.load_img(
        'E:/Study_Files/Projects/Machine_Learning/Datasets/dataset/single_prediction/cat_or_dog_1.jpg', target_size=[64,64])
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0) # Adding a fake dimension for predict method to take it as input

result = cnn.predict(test_img)

# print(train_set.class_indices)

if result[0][0] == 0:
# result[0] ----> refers to the batch, as the inputs to the  predict method are in form of batches
# result[0][0] ---------> Refers to 0th element of the 0th batch of result variable
        prediction = 'It is a DOG !'
else:
        prediction = "It is a CAT !"

print(prediction)