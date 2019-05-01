import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, Normalizer,MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import svm
from sklearn import naive_bayes

import seaborn as sns


def read_data(file):
    df = pd.read_csv(file, delimiter='|', encoding='ascii');
    return df


def fraud_instances(data_frame):
    return data_frame[data_frame.fraud == 1]


def normal_instances(data_frame):
    return data_frame[data_frame.fraud == 0]


def fraud_instances_groupby(fraud_df, column_name):
    return fraud_df.groupby(column_name)


def labeled_data(df):
    X_train = df.iloc[:, 0:9]
    Y_train = df.iloc[:, 9]
    # trustLevels=df.iloc[:,0]
    # levels=[1,2,3,4,5,6]
    # onehot_encoder = OneHotEncoder(sparse=False)
    # one_hot_trustLevels=onehot_encoder.fit_transform(trustLevels.values.reshape(-1,1))
    # correlation(X_train)
    X_train=one_hot_trust_level(X_train)
    # print(X_train.iloc[1,:])
    return X_train, Y_train


def one_hot_trust_level(X_train):
    X_train = pd.concat([pd.get_dummies(X_train['trustLevel'], prefix='trustLevel'), X_train], axis=1, join='inner')
    X_train.drop(['trustLevel'], axis=1, inplace=True)
    return X_train


def correlation(X):
    corr = X.corr()
    sns.heatmap(corr)
    plt.show()


def scale(X, scaler):
    #X['totalScanTimeInSeconds'] = X['totalScanTimeInSeconds'].values.reshape(-1, 1)
    #X['grandTotal'] = X['grandTotal'].values.reshape(-1, 1)
    #X['lineItemVoids'] = X['lineItemVoids'].values.reshape(-1, 1)
    #X['scansWithoutRegistration'] = X['scansWithoutRegistration'].values.reshape(-1, 1)
    #X['quantityModifications'] = X['quantityModifications'].values.reshape(-1, 1)
    return scaler.transform(X)


def normalize(X):
    return Normalizer().fit_transform(X)


df = read_data('../DMC_2019_task/train.csv')
fraud_df = fraud_instances(df)
groupByTrustLevel = fraud_instances_groupby(fraud_df, 'trustLevel')


# print(list(groupByTrustLevel)[1])
# print(groupByTrustLevel.count())
# print(fraud_df.head())
# print(fraud_df.count())
# print(fraud_df.iloc[2,])

# target_count=df.groupby(['fraud','trustLevel']).count()
# print(target_count)
# target_count.plot(kind='bar')
# plt.show()
# vc=df.fraud.value_counts()
# vc.plot(kind='bar')
# plt.show()
# print(scale(training_data(df)[0],MinMaxScaler()))

def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


def oversampling(X, y):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(k_neighbors=30, ratio='minority')
    X_sm, y_sm = smote.fit_resample(X, y)
    # print(len(y_sm)-len(y))
    # plot_2d_space(X_sm, y_sm, 'Balanced dataset (2 PCA components)')
    return X_sm, y_sm


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test
