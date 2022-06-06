from utility.dataset import Dataset
from utility.constant import SEED, SEQUENCE_LENGHT, VERBOSE
from utility.plot import save_correlation
from model import GraphConvModel
from datetime import date
import os
import random
import numpy as np
import tensorflow as tf
from correlation import granger_causality, cosine_similarity, dtw, dummy_correlation, pearson_correlation


tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

if __name__ == '__main__':

    dataset = Dataset('data/train.csv')

    train_date = (date(2013, 1, 1), date(2015, 1, 1))
    val_date = (date(2015, 1, 1), date(2016, 1, 1))
    test_date = (date(2016, 1, 1), date(2016, 12, 31))

    corr_functions = [
        granger_causality,
        pearson_correlation,
        dummy_correlation,
        dtw,
        cosine_similarity,

    ]

    for idx, corr_function in enumerate(corr_functions):
        print()
        print(f'   TEST #{idx}   '.center(78, '='))
        print(f'- correlation method: {corr_function.__name__} \n')

        print('Generating features')

        train = []
        if corr_function.__name__ == 'granger_causality':
            for i in range(3):
                train.append(np.load('numpy data/train' + str(i) + '.npy'))
            train = tuple(train)
        else:
            train = dataset.get_features(
                train_date[0],
                train_date[1],
                corr_function
            )

        # for i, elem in enumerate(train):
        #     np.save('numpy data/train' + str(i), elem)

        save_correlation(train[1][0], corr_function.__name__ + '_train')

        validation = []
        if corr_function.__name__ == 'granger_causality':
            for i in range(3):
                validation.append(np.load('numpy data/validation' + str(i) + '.npy'))
            validation = tuple(validation)
        else:
            validation = dataset.get_features(
                val_date[0],
                val_date[1],
                corr_function
            )

        # for i, elem in enumerate(validation):
        #     np.save('numpy data/validation' + str(i), elem)

        save_correlation(validation[1][0], corr_function.__name__ + '_val')

        test = []
        if corr_function.__name__ == 'granger_causality':
            for i in range(3):
                test.append(np.load('numpy data/test' + str(i) + '.npy'))
            test = tuple(test)
        else:
            test = dataset.get_features(
                test_date[0],
                test_date[1],
                corr_function
            )

        # for i, elem in enumerate(test):
        #     np.save('numpy data/test' + str(i), elem)

        save_correlation(validation[1][0], corr_function.__name__ + '_test')

        model = GraphConvModel(
            SEQUENCE_LENGHT,
            corr_function.__name__
        )
        model.train(train, validation)
        # model.save('models/' + corr_function.__name__)
        y_pred, y = model.predict(test)
        model.plot_predictions(y, y_pred, 7, 15)
