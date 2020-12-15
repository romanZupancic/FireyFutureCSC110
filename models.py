"""
All model architecture and training loops are defined in this file.

Copyright and Usage Information
===============================
This file is Copyright (c) 2020 Daniel Hocevar and Roman Zupancic.

This files contents may not be modified or redistributed without written
permission from Daniel Hocevar and Roman Zupancic
"""

from typing import List, Optional, Tuple, Any
import statistics
import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import assemble_data

def train_all_models() -> None:
    """
    Train all three models in this file (the SVM, the Artificial Neural Network,
    and the Recurrent Neural Network (DLSTM)).

    Only the two neural networks are saved in the models directory.
    The svm must be trained every time.
    """
    print('Starting svm training.')
    svm_model, svm_test_x, svm_test_y = train_svm()
    print('SVM training done, testing SVM accuracy.')
    svm_accuracy = test_accuracy(svm_model, svm_test_x, svm_test_y)
    print(f'Final SVM Accuracy: {svm_accuracy * 100}%')
    print('Starting ANN training.')
    train_ann()
    print('ANN training done, testing ANN accuracy.')
    ann_accuracy = load_ann()
    print(f'Final ANN Accuracy: {ann_accuracy * 100}%')
    print('Starting DLSTM training.')
    train_dlstm()
    print('DLSTM training done, testing DLSTM accuracy.')
    dlstm_accuracy = load_dlstm()
    print(f'Final DLSTM Accuracy: {dlstm_accuracy * 100}%')
    print('Training Done!')


def train_svm() -> Tuple[svm.SVC, List, List]:
    """
    Train the svm model
    """
    # Get the input data
    train, test = svm_train_test_split()

    train_x, train_y = train
    test_x, test_y = test

    # Use scikit's built in SVC model
    model = svm.SVC()
    # Train the model
    model.fit(train_x, train_y)
    return (model, test_x, test_y)


def build_ann() -> Sequential:
    """
    Define the ann model architecture using the the keras Sequential API
    """
    # A standard neural network
    model = Sequential()
    # The 42 nodes of input (21 weather data, 21 precipitation data)
    model.add(Dense(42))
    # Three layers of dense nodes
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    # One output node
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model for use
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_ann() -> None:
    """
    Train the ann model using the standard tensorflow training loop

    Save a .h5 file containing the trained model
    """
    # Build the model
    model = build_ann()
    # Separate the training and test data
    train, test = ann_train_test_split()
    # Further separate the data to inputs and outputs
    train_x, train_y = train
    test_x, test_y = test
    # Setup a tensorboard callback, so the training can be monitored on tensorboard
    tensorboard_log = tf.keras.callbacks.TensorBoard(log_dir='./data/models/logs')
    # Train the model
    model.fit(train_x, train_y, batch_size=64, epochs=13, validation_data=(test_x, test_y),
              callbacks=[tensorboard_log])
    model.save('./data/models/ann_final.h5')


def load_ann() -> float:
    """
    Load the trained ann model and test its accuracy
    """
    test = ann_train_test_split()[1]
    test_x, test_y = test
    model = tf.keras.models.load_model('./data/models/ann_final.h5')
    model.summary()
    predictions = model.predict(test_x)
    return statistics.mean([int(int(predictions[i] + 0.5) == test_y[i])
                            for i in range(len(test_y))])


def load_dlstm() -> float:
    """
    Load the trained dlstm model and test its accuracy
    """
    testing = dlstm_train_test_split()[1]

    # Further split the data fors testing
    test_temp, test_precip, test_y = testing

    # Convert the data to numpy arrays
    test_x = {'TEMPERATURE': np.array(test_temp), 'PRECIPITATION': np.array(test_precip)}
    model = tf.keras.models.load_model('./data/models/dlstm_final.h5')
    model.summary()
    predictions = model.predict(test_x)
    return statistics.mean([int(int(predictions[i] + 0.5) == test_y[i])
                            for i in range(len(test_y))])


def build_dlstm() -> tf.keras.Model():
    """
    Define the dlstm model architecture using the tensorflow functional API
    """
    # The two inputs to the function
    precipitation_input = tf.keras.Input(shape=(21, 1), name='PRECIPITATION')
    temperature_input = tf.keras.Input(shape=(21, 1), name='TEMPERATURE')

    # The LSTM layers. These are the "recurrent" part of the neural network
    precipitation_lstm = tf.keras.layers.LSTM(48, return_sequences=False)(precipitation_input)
    temperature_lstm = tf.keras.layers.LSTM(48, return_sequences=False)(temperature_input)

    # Combine the two LSTM layers (and by extension their inputs) into
    # a single shape
    concatenated = tf.keras.layers.concatenate([precipitation_lstm, temperature_lstm])

    # Process the combined data and draw more conclusions
    dense = tf.keras.layers.Dense(100, activation='relu')(concatenated)

    # Produce a single output
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense)

    # Use the inputs and outputs to build the model
    model = tf.keras.Model(inputs=[precipitation_input, temperature_input], outputs=[output],)

    # Print a table of the model layers
    model.summary()

    # Compile the model, making it functional
    model.compile(optimizer='adam', loss={'output': 'binary_crossentropy'}, metrics=['accuracy'])

    return model


def train_dlstm() -> None:
    """
    Train the dlstm model using the standard tensorflow training loop

    Save a .h5 file containing the trained model
    """
    # Build the model
    model = build_dlstm()

    # Get the split test and training data
    training, testing = dlstm_train_test_split()

    # Further split the data for training and testing
    train_temp, train_precip, train_y = training
    test_temp, test_precip, test_y = testing

    # Convert the data to numpy arrays
    train_x = {'TEMPERATURE': np.array(train_temp), 'PRECIPITATION': np.array(train_precip)}
    test_x = {'TEMPERATURE': np.array(test_temp), 'PRECIPITATION': np.array(test_precip)}

    # Setup an early stopping (to prevent overfitting)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # Setup tensorboard for monitoring
    tensorboard_log = tf.keras.callbacks.TensorBoard(log_dir='./data/models/logs')

    # Train the model
    model.fit(x=train_x, y={'output': np.array(train_y)},
              validation_data=(test_x, np.array(test_y)),
              batch_size=64, epochs=60, callbacks=[tensorboard_log, early_stop])
    model.save('./data/models/dlstm_final.h5')


def svm_train_test_split() -> Tuple[List, List]:
    """
    Create a train set and a test set using the svm training data
    """
    svm_data = assemble_data.read_svm_training_data()
    # Randomize the data
    svm_data = pd.DataFrame(shuffle(np.array(svm_data), random_state=0),
                            columns=['TEMPERATURE', 'PRECIPITATION', 'FIRE'])
    # Convert the data to a np.array
    svm_x = np.array(pd.DataFrame({'TEMPERATURE': svm_data['TEMPERATURE'],
                                   'PRECIPITATION': svm_data['PRECIPITATION']}))
    svm_y = np.array(svm_data['FIRE'])

    # Split the data between test data and training data
    train_x = svm_x[:26000]
    train_y = svm_y[:26000]
    test_x = svm_x[26000:]
    test_y = svm_y[26000:]
    return ([train_x, train_y], [test_x, test_y])


def ann_train_test_split(train_percent: Optional[int] = 80) -> Tuple[List, List]:
    """
    Create a train set and a test set using the ann training data
    """
    if train_percent > 100:
        raise ValueError('You are trying to train more than 100% of your data.')

    ann_data = assemble_data.read_ann_training_data()
    ann_data = shuffle(ann_data, random_state=0)
    ann_x = list(ann_data['WEATHER'])
    ann_y = list(ann_data['FIRE'])

    alloted_train = int(len(ann_data) * train_percent // 100)

    # Split the data between training and testing
    train_x = ann_x[:alloted_train]
    train_y = ann_y[:alloted_train]
    test_x = ann_x[alloted_train:]
    test_y = ann_y[alloted_train:]

    return ([train_x, train_y], [test_x, test_y])


def dlstm_train_test_split(train_percent: Optional[int] = 80) -> Tuple[List, List]:
    """
    Create a train set and a test set using the dlstm training data
    """
    if train_percent > 100:
        raise ValueError('You are trying to train more than 100% of your data')

    dlstm_data = assemble_data.read_dlstm_training_data()
    # Randomize the data
    dlstm_data = shuffle(dlstm_data, random_state=0)
    temp_x = list(dlstm_data['TEMPERATURE'])
    precip_x = list(dlstm_data['PRECIPITATION'])
    ann_y = list(dlstm_data['FIRE'])

    # Calculate the amount of data to split
    alloted_train = int(len(dlstm_data) * train_percent // 100)

    # Split the data between training and testing data
    temp_train_x = temp_x[:alloted_train]
    precip_train_x = precip_x[:alloted_train]
    train_y = ann_y[:alloted_train]

    temp_test_x = temp_x[alloted_train:]
    precip_test_x = precip_x[alloted_train:]
    test_y = ann_y[alloted_train:]
    return ([temp_train_x, precip_train_x, train_y], [temp_test_x, precip_test_x, test_y])


def test_accuracy(model: Any, test_x: List, test_y: List) -> float:
    """
    Test the accuracy of the input model on the input test sets
    """
    predictions = model.predict(test_x)
    return statistics.mean([int(predictions[i] == test_y[i]) for i in range(len(test_y))])


if __name__ == '__main__':
    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': ['pandas', 'sklearn', 'datetime', 'assemble_data', 'sklearn',
    #                       'statistics', 'tensorflow', 'numpy', 'random', 'typing'],
    #     'allowed-io': [],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200']
    # })

    should_build = input("Are you sure you want to build all data? (Y/n): ")

    if should_build == 'Y':
        train_all_models()
