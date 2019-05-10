import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from preprocessing import *


def plot_2d_space(labeled_X, labeled_y, unlabeled_X, label='Classes'):
    colors = ['blue', 'red']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(labeled_y), colors, markers):
        plt.scatter(
            labeled_X[labeled_y == l, 0],
            labeled_X[labeled_y == l, 1],
            c=c, label=l, marker=m
        )

    plt.scatter(unlabeled_X[:, 0], unlabled_X[:, 1], c='green', marker='*', label=2)
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

def plot_3d_space(labeled_X, labeled_y, unlabeled_X, label='Classes'):
    from mpl_toolkits.mplot3d import Axes3D
    colors = ['blue', 'red']
    markers = ['o', 's']
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    for l, c, m in zip(np.unique(labeled_y), colors, markers):
        ax.scatter(
            labeled_X[labeled_y == l, 0],
            labeled_X[labeled_y == l, 1],
            #labeled_X[labeled_y == l, 2],
            c=c, label=l, marker=m
        )

    ax.scatter(unlabeled_X[:, 0], unlabled_X[:, 1],unlabled_X[:, 2], c='green', marker='*', label=2)
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

labeled_df = read_data('../DMC_2019_task/train.csv')
unlabled_df=read_data('../DMC_2019_task/test.csv')
labeled_X, labeled_y = labeled_data(labeled_df)
unlabled_X =one_hot_trust_level(unlabled_df)
X=pd.concat([labeled_X,unlabled_X])
scaler=StandardScaler()
scaler.fit(X)
labeled_X= scale(labeled_X, scaler)
unlabled_X=scale(unlabled_X,scaler)
pca = PCA(n_components=3)
#tsne=TSNE(n_components=3)
pca_labeled_X=pca.fit_transform(labeled_X)
pca_unlabeled_X=pca.fit_transform(unlabled_X)
plot_2d_space(pca.fit_transform(labeled_X), labeled_y, pca.fit_transform(unlabled_X), 'PCA 3 components')
import seaborn as sns
#plt.style.use('seaborn')
# 2D density plot:
#sns.kdeplot(pca_labeled_X[:,0], pca_labeled_X[:,0], cmap="Reds", shade=True)
#sns.kdeplot(pca_unlabeled_X[:,0], pca_unlabeled_X[:,0], cmap="Blues", shade=True)
#plt.title('Overplotting? Try 2D density graph', loc='left')
#plt.show()
