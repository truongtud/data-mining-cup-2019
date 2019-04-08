from preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import naive_bayes
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import  Sequential,layers
import tensorflow as tf
import tensorflow.keras.metrics as kMetrics
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import tensorflow.keras.backend as K
from custom_metrics import  CustomMetrics




df = read_data('../DMC_2019_task/train.csv')
fraud_df = fraud_instances(df)
groupByTrustLevel = fraud_instances_groupby(fraud_df, 'trustLevel')

X, y = training_data(df)
X = scale(X, StandardScaler())
# pca = PCA(n_components=2)
# X = pca.fit_transform(X)
# plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')
X_sm, y_sm = oversampling(X, y)
X_train, X_test, y_train, y_test = split_data(X_sm, y_sm)
count_fraud = 0
for i in range(len(y_test)):
    if y_test[i] == 1:
        count_fraud += 1


# print(count_fraud)

class ModelFactory:

    @staticmethod
    def createDecisionTree():
        return DecisionTreeClassifier()

    @staticmethod
    def createSVM():
        return svm.SVC(kernel='linear', C=0.5, degree=3)

    @staticmethod
    def creatLogistic():
        return LogisticRegression()


feature_cols = ['trustLevel', 'totalScanTimeInSeconds', 'grandTotal','lineItemVoids', 'scansWithoutRegistration',
                'quantityModifications', 'scannedLineItemsPerSecond', 'valuePerSecond', 'lineItemVoidsPerPosition']


def decisionTreeClassifier():
    dtc = ModelFactory.createDecisionTree()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    DT_cm = metrics.confusion_matrix(y_test, y_pred)
    # print("Confusion matrix using Decision Tree:", DT_cm.ravel())
    print("Confusion matrix using Decision Tree:", DT_cm)
    print("DT scores:", DT_cm[1][1] * 5 - DT_cm[0][1] * 25 - DT_cm[1][0] * 5)
    dot_data = StringIO()
    export_graphviz(dtc, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('fraud_retailer.png')
    Image(graph.create_png())


def svmClassifier():
    SVM = ModelFactory.createSVM()
    SVM.fit(X_train, y_train)
    y_SVM_pred = SVM.predict(X_test)
    SVM_cm = metrics.confusion_matrix(y_test, y_SVM_pred)
    print("Confusion matrix using SVM", SVM_cm.ravel())
    print("SVM scores:", SVM_cm[1][1] * 5 - SVM_cm[0][1] * 25 - SVM_cm[1][0] * 5)


def logisticClassifier():
    lc = ModelFactory.creatLogistic()
    lc.fit(X_train, y_train)
    y_lc_pred = lc.predict(X_test)
    lc_cm = metrics.confusion_matrix(y_test, y_lc_pred)
    print("Confusion matrix using Logistic", lc_cm.ravel())
    print("Logistic scores:", lc_cm[1][1] * 5 - lc_cm[0][1] * 25 - lc_cm[1][0] * 5)


def votingClassifier():
    votingClassifier = VotingClassifier(estimators=[
        ('dc', ModelFactory.createDecisionTree()), ('svm', ModelFactory.createSVM()),
        ('lg', ModelFactory.creatLogistic())
    ], voting='hard')
    votingClassifier.fit(X_train, y_train)
    y_voted = votingClassifier.predict(X_test)
    voting_cm = metrics.confusion_matrix(y_test, y_voted)
    print("Confusion matrix using Voting", voting_cm.ravel())
    print("Voting scores:", voting_cm[1][1] * 5 - voting_cm[0][1] * 25 - voting_cm[1][0] * 5)

def scores(y_test,y_pred):
    cf_matrix=K.variable(metrics.confusion_matrix(y_test,y_pred))
    print(cf_matrix.shape)
    return K.variable(np.array([[(cf_matrix[1][1]*5-cf_matrix[0][1]*25-cf_matrix[1][0]*5)/len(y_test)]]))




def deepNN():
    custom_metrics=CustomMetrics()
    model=Sequential()
    model.add(layers.Dense(units=8,input_dim=X_train.shape[1],activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=6,activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=6,activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=4,activation='relu'))
   # model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=4,activation='relu'))
   # model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=2,activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=2,activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy')
    model.fit(X_train,y_train,batch_size=4,epochs=200,validation_data=[X_test,y_test],callbacks=[custom_metrics])
    for k in custom_metrics.scores:
        print(k)

def cnn():
    pass

def deepAutoEncoder():
    pass


if __name__ == '__main__':
    decisionTreeClassifier()
    #svmClassifier()
    #logisticClassifier()
    #votingClassifier()
    deepNN()
