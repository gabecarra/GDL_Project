import pandas as pd
from datetime import timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from utility.constant import SEQUENCE_LENGHT, VERBOSE

# set seaborn
sns.set_theme(style='ticks')


class Dataset:
    def __init__(self, path):
        if VERBOSE:
            print(f'Reading dataset from {path}')
        self.df = pd.read_csv(path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['id'] = self.df['item'].astype(str) +\
            '_' + self.df['store'].astype(str)
        self.df.set_index(['id', 'date'], inplace=True)
        self.df.drop(['store', 'item'], axis=1, inplace=True)
        self.df = self.df.astype(float).unstack()
        self.df.columns = self.df.columns.get_level_values(1)
        self.df.sort_index(inplace=True)
        # self.df.apply(remove_periodicity, axis=1, raw=True)
        if VERBOSE:
            print(f'Dataset shape:  {self.df.shape}')
            print(self.df.head())

    def get_df(self):
        """
        Getter for the dataframe obj
        :return: pandas dataframe
        """
        return self.df.copy()

    def __create_features(self, initial_date, correlation_fn):
        """
        TODO
        :param initial_date:
        :param period:
        :return:
        """

        # rows representing store_product, columns defined by starting
        # date, through sequence_length
        all_sequence = self.df[
            pd.date_range(
                initial_date - timedelta(days=SEQUENCE_LENGHT),
                periods=SEQUENCE_LENGHT,
                freq='D')
        ].values

        # reshape in order to have (products, stores, dates)
        group_store = all_sequence.reshape((-1, 10, SEQUENCE_LENGHT))

        store_corr = np.stack([correlation_fn(i) for i in group_store], axis=0)

        # group_store = np.transpose(group_store, (0, 2, 1))

        return group_store, store_corr

    def get_features(self, start, end, correlation_fn):
        """
        TODO
        :param start:
        :param end:
        :param correlation_fn:
        :return:
        """
        x_seq, x_cor, y = [], [], []

        for date in tqdm(
                pd.date_range(start + timedelta(days=SEQUENCE_LENGHT), end)
        ):
            cur_seq, cur_corr = self.__create_features(
                date,
                correlation_fn
            )

            # x sequence, correlation and features
            x_seq.append(cur_seq)
            x_cor.append(cur_corr)

            # add the labels
            y.append(self.df[pd.to_datetime(start)].values.reshape((-1, 10)))

        # create numpy arrays with right type for x and y
        x_seq = np.concatenate(x_seq, axis=0).astype('float16')
        x_cor = np.concatenate(x_cor, axis=0).astype('float16')
        y = np.concatenate(y, axis=0).astype('float16')

        if VERBOSE:
            print(x_seq.shape, x_cor.shape, y.shape)

        return x_seq, x_cor, y

    def plot_timespan(self, initial_date, period, idx):
        """
        Given an initial date, a period of time in days, and an idx for
        the rows, plot the sales sequence starting from initial_date
        until period, for the number of products_stores defined in idx
        :param initial_date: datetime obj
        :param period: integer representing the number of days
        :param idx: tuple representing the (start, end) rows
        :return: None
        """
        plot_df = self.df[
            pd.date_range(
                initial_date - timedelta(days=period),
                periods=period,
                freq='D')
        ]
        plot_df[idx[0]:idx[1]].T.plot(figsize=(14, 5))
        plt.ylabel('sales')
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.show()

    def fit_polynomial(self, initial_date, period=None):
        temp = self.df
        if period:
            temp = self.df[
                pd.date_range(
                    initial_date - timedelta(days=period),
                    periods=period,
                    freq='D')
            ]
        temp = temp.iloc[0]
        X = [i % 365 for i in range(0, len(temp))]
        y = temp.values
        degree = 4
        coef = np.polyfit(X, y, degree)
        print('Coefficients: %s' % coef)
        # create curve
        curve = list()
        for i in range(len(X)):
            value = coef[-1]
            for d in range(degree):
                value += X[i] ** (degree - d) * coef[d]
            curve.append(value)

        values = temp.values
        diff = list()
        for i in range(len(values)):
            value = values[i] - curve[i]
            diff.append(value)
        # plot curve over original data
        plt.figure(1, figsize=(14, 5))
        plt.plot(temp.T.values)
        plt.ylabel('sales')
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.plot(curve, color='red', linewidth=3)

        plt.figure(2, figsize=(14, 5))
        plt.plot(diff)
        plt.show()


def remove_periodicity(y, degree=4):
    X = [i % 365 for i in range(0, len(y))]
    coef = np.polyfit(X, y, degree)
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i] ** (degree - d) * coef[d]
        y[i] = y[i] - value
    return y


if __name__ == '__main__':
    dataset = Dataset('../data/train.csv')
    # dataset.plot_timespan(date(2017, 11, 1), 30, idx=(0, 10))
    # train_date = (date(2013, 1, 1), date(2016, 1, 1))
    # val_date = (date(2016, 1, 1), date(2017, 1, 1))
    # test_date = (date(2017, 1, 1), date(2017, 4, 1))
    # train = dataset.get_features(
    #     train_date[0],
    #     train_date[1],
    #     np.corrcoef
    # )
    # dataset.fit_polynomial(date(2015, 1, 1))
    dataset.df.iloc[0].T.plot(figsize=(14, 5))
    plt.show()
    dataset.df.apply(remove_periodicity, axis=1, raw=True)
    dataset.df.iloc[0].T.plot(figsize=(14, 5))
    plt.show()

    # dataset.plot_timespan(date(2017, 11, 1), 365 * 4, idx=(0, 1))
    # dataset.plot_timespan(date(2017, 11, 1), 365 * 4, idx=(15, 16))
    pass
