import keras
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input, LSTM,\
    BatchNormalization, CuDNNLSTM
from spektral.layers import GCNConv
from spektral.layers.pooling import GlobalMaxPool
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import StandardScaler
from spektral.utils.convolution import gcn_filter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

sns.set_theme(style='ticks')


class GraphConvModel:
    def __init__(self, feature_size, method_name):
        self.batch_size = 256
        self.name = method_name
        self.opt = Adam(learning_rate=0.001)

        inp_lap = Input((10, 10))
        inp_feat = Input((10, feature_size))

        x = GCNConv(256, activation='relu')([inp_feat, inp_lap])
        x = GCNConv(128, activation='relu')([x, inp_lap])
        x = CuDNNLSTM(128, return_sequences=True)(x)
        x = CuDNNLSTM(64)(x)
        # x = LSTM(64, activation='relu', return_sequences=True)(x)
        # x = LSTM(32, activation='relu')(x)

        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        out = Dense(1)(x)

        self.model = keras.Model([inp_lap, inp_feat], out)
        self.model.compile(
            optimizer=self.opt,
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )

        self.scaler_seq = StandardScaler()

    def summary(self):
        """
        Prints the model summary
        :return: None
        """
        self.model.summary()

    def train(self, train, validation):
        """
        Given the train, validation and test data, train the model for
        each store, and plot the history
        :param train: tuple containing (sequence, correlation matrix,
        features, output) for the train data
        :param validation: tuple containing (sequence,
        correlation matrix, features, output) for the validation data
        :param test: tuple containing (sequence, correlation matrix,
        features, output) for the test data
        :return: None
        """
        x_train_seq, x_train_lap, y_train = \
            self.__preprocess_data(train)

        x_val_seq, x_val_lap, y_val = \
            self.__preprocess_data(validation)

        val_rmse = []
        rmse = []
        loss = []
        val_loss = []
        steps = []
        for store in range(10):
            print('-------', 'store', store, '-------')

            es = EarlyStopping(patience=5, verbose=0, min_delta=0.001,
                               monitor='val_loss', mode='auto',
                               restore_best_weights=True)

            history = self.model.fit(
                [x_train_lap, x_train_seq],
                y_train[:, store], epochs=100, batch_size=self.batch_size,
                validation_data=(
                    [x_val_lap, x_val_seq],
                    y_val[:, store]
                ),
                callbacks=[es],
                verbose=0
            )
            if steps:
                steps.append(
                    steps[-1] + len(history.history['root_mean_squared_error'])
                )
            else:
                steps.append(len(history.history['root_mean_squared_error']))
            rmse = rmse + history.history['root_mean_squared_error']
            val_rmse = val_rmse + history.history['val_root_mean_squared_error']
            loss = loss + history.history['loss']
            val_loss = val_loss + history.history['val_loss']

        self.__plot_history(rmse, val_rmse, loss, val_loss, steps)

    def predict(self, test):
        """
        Given the true test data tuple, predicts using the trained
        model, and returns both the prediction and the true results
        :param test: tuple containing (sequence, correlation matrix,
        features, output) for the test data
        :return: prediction and true values
        """
        x_test_seq, x_test_lap, y_test = \
            self.__preprocess_data(test)

        pred_test_all = np.zeros(y_test.shape)
        for store in range(10):

            pred_test_all[:, store] = self.model.predict(
                [x_test_lap, x_test_seq]).ravel()

        y_pred_test = self.scaler_seq.inverse_transform(pred_test_all)

        # save prediction results
        columns = ['store_' + str(i) for i in range(1, 11)]
        pred_df = pd.DataFrame(y_pred_test, columns=columns)
        pred_df.to_csv("results/predictions/" + self.name + '.csv')

        y_test = self.scaler_seq.inverse_transform(y_test)

        self.__calculate_error(y_test, y_pred_test)

        return y_pred_test, y_test

    def __calculate_error(self, y_true, y_pred):
        """
        given the true values and the predictions, calculate the mse and
        plots it in a bar diagram
        :param y_true: matrix containing the true prices
        :param y_pred: matrix containing the predicted prices
        :return: None
        """
        error = {}
        tot_error = 0
        for store in range(10):
            error[store] = np.sqrt(mean_squared_error(y_true[:, store],
                                                      y_pred[:, store]))
            print(f'Store {store} MSE: {error[store]}')
            tot_error += error[store]

        json_data = error.copy()
        json_data['tot_error'] = tot_error / 10
        print(f'Tot MSE: {json_data["tot_error"]}')
        with open('results/MSE/' + self.name + '.json', 'w') as outfile:
            json.dump(json_data, outfile)

        plt.figure(figsize=(14, 5))
        plt.bar(range(10), error.values())
        plt.xticks(range(10), ['store_' + str(s) for s in range(10)])
        plt.ylabel('errors')
        np.set_printoptions(False)
        plt.savefig('results/plots/errors/' + self.name + '.pdf')

    def plot_predictions(self, y_true, y_pred, store, item):
        """
        Given the predictions, the true values, a store index, and an
        item index, plot the prediction through time
        :param y_true: matrix containing the true prices
        :param y_pred: matrix containing the predicted prices
        :param store: store index (from 0 to 9)
        :param item: item index (from 0 to 49)
        :return:
        """
        y_true = y_true.reshape(50, -1, 10)
        y_pred = y_pred.reshape(50, -1, 10)

        plt.figure(figsize=(11, 5))
        plt.plot(y_true[item, :, store], label='true')
        plt.plot(y_pred[item, :, store], label='prediction')
        plt.title(f"store: {store} item: {item}")
        plt.legend()
        plt.ylabel('sales')
        plt.xlabel('date')
        plt.savefig('results/plots/predictions/' + self.name + '.pdf')

    def __plot_history(self, rmse, val_rmse, loss, val_loss, steps):

        fig, axs = plt.subplots(2, figsize=(15, 10))

        # summarize history for accuracy
        axs[0].plot(rmse)
        axs[0].plot(val_rmse)
        axs[0].vlines(
            x=steps,
            ymin=min(min(rmse), min(val_rmse)),
            ymax=max(max(rmse), max(val_rmse)),
            colors='green',
            ls=':',
            lw=2,
            alpha=0.5
        )

        axs[0].set_title('model root_mean_squared_error')
        axs[0].set(xlabel='epoch', ylabel='root_mean_squared_error')
        axs[0].legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        axs[1].plot(loss)
        axs[1].plot(val_loss)
        axs[1].vlines(
            x=steps,
            ymin=min(min(loss), min(val_loss)),
            ymax=max(max(loss), max(val_loss)),
            colors='green',
            ls=':',
            lw=2,
            alpha=0.5
        )
        axs[1].set_title('model loss')
        axs[1].set(xlabel='epoch', ylabel='loss')
        axs[1].legend(['train', 'test'], loc='upper left')

        plt.savefig('results/plots/history/' + self.name + '.pdf')

    def __preprocess_data(self, data):
        x_seq, x_cor, y = data

        # preprocess data with mean 0 and std 1
        x_seq = self.scaler_seq.fit_transform(
            x_seq.reshape(-1, 10)
        ).reshape(x_seq.shape)

        y = self.scaler_seq.transform(y)

        # laplacian preprocessing for graph convolutional layer
        x_cor = gcn_filter(x_cor)

        return x_seq, x_cor, y

    def save(self, path):
        """
        TODO
        :param path:
        :return:
        """
        self.model.save(path)


if __name__ == '__main__':
    model = GraphConvModel(14, 7, 'test')
    model.summary()
