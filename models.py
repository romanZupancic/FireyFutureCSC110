import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle
import numpy as np
from typing import List
import statistics

SVM_TRAINING_DATA_PATH = './data/training_data/svm_training_data.csv'

def read_svm_training_data() -> pd.DataFrame:
    """Return a dataframe containing the data from /data/training_data/svm_training_data.csv"""
    svm_data = pd.read_csv(SVM_TRAINING_DATA_PATH)
    final_svm_data = pd.DataFrame({'TEMPERATURE': svm_data['TEMPERATURE'], 'PRECIPITATION': svm_data['PRECIPITATION'], 'FIRE': svm_data['FIRE']})
    return final_svm_data


def svm_train_test_split() -> List:
    svm_data = read_svm_training_data()
    svm_data = pd.DataFrame(shuffle(np.array(svm_data), random_state=0), columns=['TEMPERATURE', 'PRECIPITATION', 'FIRE'])
    svm_x = np.array(pd.DataFrame({'TEMPERATURE': svm_data['TEMPERATURE'], 'PRECIPITATION': svm_data['PRECIPITATION']}))
    svm_y = np.array(svm_data['FIRE'])
    train_x = svm_x[:26000]
    train_y = svm_y[:26000]
    test_x = svm_x[26000:]
    test_y = svm_y[26000:]
    return [[train_x, train_y], [test_x, test_y]]
    #x = pd.DataFrame(svm_x, columns=['TEMPERATURE', 'PRECIPITATION'])
    #x['FIRE'] = svm_y
    #assert all(np.array(x)[i][a] == np.array(svm_data)[i][a] for i in range(len(x)) for a in range(len(np.array(x)[i])))

train, test = svm_train_test_split()

train_x, train_y = train

test_x, test_y = train

model = svm.SVC(C=200)

model.fit(train_x, train_y)

def test_accuracy():
    predictions = model.predict(test_x)
    return statistics.mean([int(predictions[i] == test_y[i]) for i in range(len(test_y))])

