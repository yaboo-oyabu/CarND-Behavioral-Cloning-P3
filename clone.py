import os
import csv
import cv2
import matplotlib

import numpy as np
import random
import sklearn

from math import ceil
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

from sklearn.model_selection import train_test_split

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_samples(path_to_csv):
    """
    Generate train, validation, and test dataset.
    """
    rows = []
    
    with open(path_to_csv) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip header
        for row in reader:
            steering_center = float(row[3])
            rows.append([row[0], steering_center, False])
            rows.append([row[0], -steering_center, True]) # Flip image if the last element is True

    # Use 10% of whole data for test dataset.
    train_and_validation_samples, test_samples = train_test_split(rows, test_size=0.1)
    # Use 80% of the rest data for training dataset, and 20% for validation dataset.
    train_samples, validation_samples = train_test_split(train_and_validation_samples, test_size=0.2)

    return train_samples, validation_samples, test_samples

def preprocess(image):
    """
    Preprocess an image for ML model training.
    """
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Remove sky and sea colors
    lower = np.array([80,0,0])
    upper = np.array([140,120,255])
    mask = cv2.inRange(hsv_image, lower, upper) == 255
    hsv_image[mask] = 255

    # Use only S value
    input_image = hsv_image[:,:,1]

    # Crop image
    input_image = input_image[50:-30]

    # Resize image
    input_image = cv2.resize(input_image, (80, 20))

    return input_image

def generator(samples, batch_size=32):
    """
    Generate batches for train, validation and test dataset.
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/' + batch_sample[0].split('/')[-1]
                image = np.expand_dims(preprocess(cv2.imread(name)), -1)
                steering = float(batch_sample[1])
                if batch_sample[2]:
                    image = np.fliplr(image)
                images.append(image)
                angles.append(steering)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def model_fn():
    """
    Defines a convolutional neural network for steering prediction.
    """
    model = Sequential()

    # Input layer and normalization
    model.add(InputLayer(input_shape=(20, 80, 1)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # Convolutional layer 1
    model.add(Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Convolutional layer 2
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Dropout for regularization.
    model.add(Dropout(0.2))

    # Full connected layer
    model.add(Flatten())
    model.add(Dense(100))

    # Predicted steering
    model.add(Dense(1))
    print(model.summary())
    return model

def main():
    """
    main function for creating ML model for steering prediction.
    """
    # Set our batch size
    batch_size=32

    # Create datasets for train, validation and test.
    train_samples, validation_samples, test_samples = get_samples('data/driving_log.csv')

    # Compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    test_generator = generator(test_samples, batch_size=batch_size)
    model = model_fn()
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(
        train_generator,
        steps_per_epoch=ceil(len(train_samples)/batch_size),
        validation_data=validation_generator,
        validation_steps=ceil(len(validation_samples)/batch_size),
        epochs=20, verbose=1)

    # Save model
    model.save("results/model.h5")

    ### Print MSE for test dataset.
    result = model.evaluate_generator(test_generator, steps=50)
    print("MSE for test data: {}".format(result))

    ### Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('output/{}/output.png'.format(target))

if __name__ == "__main__":
    main()