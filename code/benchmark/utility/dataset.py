import pandas as pd
from datetime import timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from utility.constant import SEQUENCE_LENGHT, VERBOSE

# set seaborn
sns.set_theme(style='ticks')


class Dataset:
    def __init__(self, path):
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

        # reshape in order to have (stores, products, dates)
        group_store = all_sequence.reshape((-1, 10, SEQUENCE_LENGHT))

        store_corr = np.stack([correlation_fn(i) for i in group_store], axis=0)

        store_features = np.stack([
            group_store.mean(axis=2),
            group_store[:, :, int(SEQUENCE_LENGHT / 2):].mean(axis=2),
            group_store.std(axis=2),
            group_store[:, :, int(SEQUENCE_LENGHT / 2):].std(axis=2),
            skew(group_store, axis=2),
            kurtosis(group_store, axis=2),
            np.apply_along_axis(
                lambda x: np.polyfit(np.arange(0, SEQUENCE_LENGHT), x, 1)[0],
                2, group_store)
        ], axis=1)

        group_store = np.transpose(group_store, (0, 2, 1))
        store_features = np.transpose(store_features, (0, 2, 1))

        return group_store, store_corr, store_features

    def get_features(self, start, end, correlation_fn):
        """
        TODO
        :param start:
        :param end:
        :param correlation_fn:
        :return:
        """
        x_seq, x_cor, x_feat, y = [], [], [], []

        for date in tqdm(
                pd.date_range(start + timedelta(days=SEQUENCE_LENGHT), end)
        ):
            cur_seq, cur_corr, cur_feat = self.__create_features(
                date,
                correlation_fn
            )

            # x sequence, correlation and features
            x_seq.append(cur_seq)
            x_cor.append(cur_corr)
            x_feat.append(cur_feat)

            # add the labels
            y.append(self.df[pd.to_datetime(start)].values.reshape((-1, 10)))

        # create numpy arrays with right type for x and y
        x_seq = np.concatenate(x_seq, axis=0).astype('float16')
        x_cor = np.concatenate(x_cor, axis=0).astype('float16')
        x_feat = np.concatenate(x_feat, axis=0).astype('float16')
        y = np.concatenate(y, axis=0).astype('float16')

        if VERBOSE:
            print(x_seq.shape, x_cor.shape, x_feat.shape, y.shape)

        return x_seq, x_cor, x_feat, y

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


if __name__ == '__main__':
    dataset = Dataset('../data/train.csv')
    # dataset.plot_timespan(date(2017, 11, 1), 30, idx=(0, 10))
    train_date = date(2013, 1, 1)
    valid_date = date(2015, 1, 1)
    test_date = date(2016, 1, 1)
    dataset.get_features(train_date, valid_date, np.corrcoef)
    pass
