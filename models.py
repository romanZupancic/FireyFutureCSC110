import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle
import numpy as np
from typing import List, Optional, Tuple
import statistics
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

SVM_TRAINING_DATA_PATH = './data/training_data/svm_training_data.csv'
ANN_TRAINING_DATA_PATH = './data/training_data/ann_training_data.csv'
DLSTM_TRAINING_DATA_PATH = './data/training_data/dlstm_training_data.csv'

def read_svm_training_data() -> pd.DataFrame:
    """Return a dataframe containing the data from /data/training_data/svm_training_data.csv"""
    svm_data = pd.read_csv(SVM_TRAINING_DATA_PATH)
    final_svm_data = pd.DataFrame({'TEMPERATURE': svm_data['TEMPERATURE'],
                                   'PRECIPITATION': svm_data['PRECIPITATION'],
                                   'FIRE': svm_data['FIRE']})
    return final_svm_data


def read_ann_training_data() -> pd.DataFrame:
    """Return a dataframe containing the data from /data/training_data/ann_training_data.csv"""
    ann_data = pd.read_csv(ANN_TRAINING_DATA_PATH)

    weather_total = []
    for _, weather in ann_data['WEATHER'].iteritems():
        weather_total.append([float(item) for item in weather.lstrip('[').rstrip(']').split(', ')])
    final_ann_data = pd.DataFrame({'WEATHER': weather_total, 'FIRE': ann_data['FIRE']})
    return final_ann_data


def read_dlstm_training_data() -> pd.DataFrame:
    """Return a dataframe containing the data from /data/training_data/rnn_training_data.csv"""
    dlstm_data = pd.read_csv(DLSTM_TRAINING_DATA_PATH)
    weather_total = {'TEMPERATURE': [], 'PRECIPITATION': []}
    for _, weather in dlstm_data.iterrows():
        for heading in ['TEMPERATURE', 'PRECIPITATION']:
            weather_total[heading].append([float(item) for item in weather[heading].lstrip('[').rstrip(']').split(', ')])
    final_dlstm_data = pd.DataFrame({'TEMPERATURE': weather_total['TEMPERATURE'], 'PRECIPITATION': weather_total['PRECIPITATION'], 'FIRE': dlstm_data['FIRE']})
    return final_dlstm_data


def test_accuracy(model, test_x, test_y):
    predictions = model.predict(test_x)
    return statistics.mean([int(predictions[i] == test_y[i]) for i in range(len(test_y))])


def svm_train_test_split() -> Tuple[List, List]:
    svm_data = read_svm_training_data()
    svm_data = pd.DataFrame(shuffle(np.array(svm_data), random_state=0), columns=['TEMPERATURE', 'PRECIPITATION', 'FIRE'])
    svm_x = np.array(pd.DataFrame({'TEMPERATURE': svm_data['TEMPERATURE'], 'PRECIPITATION': svm_data['PRECIPITATION']}))
    svm_y = np.array(svm_data['FIRE'])
    train_x = svm_x[:26000]
    train_y = svm_y[:26000]
    test_x = svm_x[26000:]
    test_y = svm_y[26000:]
    return ([train_x, train_y], [test_x, test_y])
    #x = pd.DataFrame(svm_x, columns=['TEMPERATURE', 'PRECIPITATION'])
    #x['FIRE'] = svm_y
    #assert all(np.array(x)[i][a] == np.array(svm_data)[i][a] for i in range(len(x)) for a in range(len(np.array(x)[i])))


def ann_train_test_split(train_percent: Optional[int] = 80,
                         test_percent: Optional[int] = 20) -> Tuple[List, List]:
    if train_percent + test_percent != 100:
        raise ValueError('Your percentages do not add up to 100.')

    ann_data = read_ann_training_data()
    ann_data = shuffle(ann_data, random_state=0)
    ann_x = list(ann_data['WEATHER'])
    ann_y = list(ann_data['FIRE'])

    alloted_train = int(len(ann_data) * train_percent // 100)

    train_x = ann_x[:alloted_train]
    train_y = ann_y[:alloted_train]
    test_x = ann_x[alloted_train:]
    test_y = ann_y[alloted_train:]
    # eval_x = ann_x[alloted_train+alloted_test:]
    # eval_y = ann_y[alloted_train+alloted_test:]
    return ([train_x, train_y], [test_x, test_y])


def dlstm_train_test_split(train_percent: Optional[int] = 80,
                           test_percent: Optional[int] = 20) -> Tuple[List, List]:
    if train_percent + test_percent != 100:
        raise ValueError('Your percentages do not add up to 100.')

    dlstm_data = read_dlstm_training_data()
    dlstm_data = shuffle(dlstm_data, random_state=0)
    temp_x = list(dlstm_data['TEMPERATURE'])
    precip_x = list(dlstm_data['PRECIPITATION'])
    ann_y = list(dlstm_data['FIRE'])

    alloted_train = int(len(dlstm_data) * train_percent // 100)

    temp_train_x = temp_x[:alloted_train]
    precip_train_x = precip_x[:alloted_train]
    train_y = ann_y[:alloted_train]
    temp_test_x = temp_x[alloted_train:]
    precip_test_x = precip_x[alloted_train:]
    test_y = ann_y[alloted_train:]
    #return ([{'TEMPERATURE': temp_train_x, 'PRECIPITATION': precip_train_x}, train_y], [{'TEMPERATURE': temp_test_x, 'PRECIPITATION': precip_test_x}, test_y])
    return ([temp_train_x, precip_train_x, train_y], [temp_test_x, precip_test_x, test_y])


def train_svm() -> None:
    train, test = svm_train_test_split()

    train_x, train_y = train
    test_x, test_y = test

    model = svm.SVC()
    model.fit(train_x, train_y)


def build_ann() -> Sequential:
    model = Sequential()
    model.add(Dense(42))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_ann():
    model = build_ann()
    train, test = ann_train_test_split()
    train_x, train_y = train
    test_x, test_y = test
    tensorboard_log = tf.keras.callbacks.TensorBoard(log_dir='./data/models/logs')
    model.fit(train_x, train_y, batch_size=64, epochs=13, validation_data=(test_x, test_y), callbacks=[tensorboard_log])
    model.save('./data/models/ann_v3.h5')


def load_ann():
    train, test = ann_train_test_split()
    train_x, train_y = train
    test_x, test_y = test
    model = tf.keras.models.load_model('./data/models/ann_v3.h5')
    model.summary()
    predictions = model.predict(test_x)
    return statistics.mean([int(int(predictions[i] + 0.5) == test_y[i]) for i in range(len(test_y))])


def mini_lstm():
    model = Sequential()
    model.add(tf.keras.layers.LSTM(1))
    model.compile(optimizer='adam', loss='binary_crossentropy')


def build_dlstm() -> tf.keras.Model():
    precipitation_input = tf.keras.Input(shape=(21, 1), name='PRECIPITATION')
    temperature_input = tf.keras.Input(shape=(21, 1), name='TEMPERATURE')

    precipitation_lstm = tf.keras.layers.LSTM(48, return_sequences=False)(precipitation_input)
    temperature_lstm = tf.keras.layers.LSTM(48, return_sequences=False)(temperature_input)

    concatenated = tf.keras.layers.concatenate([precipitation_lstm, temperature_lstm])

    dense = tf.keras.layers.Dense(100, activation='relu')(concatenated)

    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense)

    model = tf.keras.Model(inputs=[precipitation_input, temperature_input], outputs=[output],)

    model.summary()

    model.compile(optimizer='adam', loss={'output': 'binary_crossentropy'},metrics=['accuracy'])

    return model


def train_dlstm() -> None:
    model = build_dlstm()

    training, testing = dlstm_train_test_split()

    train_temp, train_precip, train_y = training
    test_temp, test_precip, test_y = testing

    train_x = {'TEMPERATURE': np.array(train_temp), 'PRECIPITATION': np.array(train_precip)}
    # train_x = {'TEMPERATURE': train_temp, 'PRECIPITATION': train_precip}
    test_x = {'TEMPERATURE': np.array(test_temp), 'PRECIPITATION': np.array(test_precip)}
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tensorboard_log = tf.keras.callbacks.TensorBoard(log_dir='./data/models/logs')
    model.fit(x=train_x, y={'output':np.array(train_y)}, validation_data=(test_x, np.array(test_y)), batch_size=64, epochs=60, callbacks=[tensorboard_log, early_stop])
    model.save('./data/models/dlstm_final.h5')

train_dlstm()