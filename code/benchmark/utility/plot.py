import matplotlib.pyplot as plt
import seaborn
import glob
import pandas as pd
import json
import numpy as np

seaborn.set()


def plot_train_val(save=False):
    for method_path in glob.glob('../logs/*/'):
        print(method_path.split('/')[-2])
        metrics = None
        for csv_path in glob.glob(method_path + 'version_*/metrics.csv'):
            df = pd.read_csv(csv_path)
            val_df = df[df.columns[:6]].dropna().reset_index()
            train_df = df[df.columns[8:14]].dropna().reset_index()
            df = pd.concat([train_df, val_df], axis=1).drop(['index'], axis=1)
            if metrics is None:
                metrics = df
            else:
                metrics += df
        metrics /= 5
        for i, col in enumerate(['loss', 'mae', 'mape']):
            plt.figure(i)
            plt.plot(metrics['train_' + col], label="train")
            plt.plot(metrics['val_' + col], label="validation")
            plt.ylim(0, 0.2) if col == 'mape' else plt.ylim(2.8, 3.8)
            plt.xlim(-0.5)
            plt.xlabel('Epoch')
            plt.ylabel(col.title())
            plt.legend()
            if save:
                plt.savefig('plots/' + method_path.split('/')[-2] + '_' + col + '.pdf')
            else:
                plt.show()
            plt.close(i)
        print('train loss:', metrics['train_loss'].min())
        print('val loss:', metrics['val_loss'].min())


def print_test():
    for method_path in glob.glob('../results/*/'):
        print(method_path.split('/')[-2])
        metrics = dict(test_mae=0, test_mape=0, test_loss=0)
        for json_path in glob.glob(method_path + 'version_*/res.json'):
            f = open(json_path, "r")
            data = json.loads(f.read())
            for key in metrics.keys():
                metrics[key] += data[key]
        for key in metrics.keys():
            metrics[key] /= 5
        print(metrics)


def plot_adj(save=False):
    for method_path in glob.glob('../numpy data/*'):
        print(method_path.split('/')[-1][:-4])
        adj = np.load(method_path)
        print(f'min: {adj.min()}, max: {adj.max()}, mean:{adj.mean()}, std: {adj.std()}')
        plt.matshow(adj)
        plt.title(method_path.split('/')[-1][:-4])
        if save:
            plt.savefig(
                'plots/' + method_path.split('/')[-1][:-4] + '_adj.pdf')
        else:
            plt.show()


if __name__ == '__main__':
    print_test()
    pass
