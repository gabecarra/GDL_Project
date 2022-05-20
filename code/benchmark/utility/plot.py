import matplotlib.pyplot as plt
import seaborn
import seaborn as sns

seaborn.set()


def save_correlation(A, fig_name):
    plt.figure()
    ax = sns.heatmap(A, cmap='viridis', linewidth=0.5)
    plt.savefig('results/plots/correlations/' + fig_name + '.pdf')
    plt.close()
