from preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import *
from sklearn import naive_bayes
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model
import tensorflow as tf
import tensorflow.keras.metrics as kMetrics
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import tensorflow.keras.backend as K
from custom_metrics import CustomMetrics
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras import regularizers
from tied_autoencoder_keras import DenseLayerAutoencoder
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

DROP_OUT_PROB = 0.2
EPOCHS = 2000
BATCH_SIZE = 128
n_folds =10

def load_labeled_data(scaler):
    df = read_data('../DMC_2019_task/train.csv')
    # fraud_df = fraud_instances(df)
    # normal_df=normal_instances(df)
    # groupByTrustLevel = fraud_instances_groupby(fraud_df, 'trustLevel')

    X, y = labeled_data(df)
    X = scale(X, scaler)
    # y=np.array(y)
    # pca = PCA(n_components=2)
    # plot_2d_space(pca.fit_transform(X), y, 'Imbalanced dataset (2 PCA components)')
    # t_sne=TSNE(n_components=2)
    # plot_2d_space(t_sne.fit_transform(X), y, 'Imbalanced dataset (2 TSNE components)')
    # X_sm, y_sm = oversampling(X, y)
    # X_sm=normalize(X_sm)
    # X_train, X_test, y_train, y_test = split_data(X, y)
    # normal_X,normal_y=training_data(normal_df)
    # normal_X_sm,normal_y_sm=oversampling(normal_X,normal_y)
    # normal_X=scale(normal_X,MinMaxScaler())

    # normal_X_train,normal_X_test,normal_y_train,normal_y_test=split_data(normal_X,normal_y)
    return X, y


def load_unlabeled_data(scaler):
    unlabled_X = scale(one_hot_trust_level(read_data('../DMC_2019_task/test.csv')), scaler)
    unlabled_X_train, unlabled_X_test, unlabled_Y_train, unlabled_Y_test = split_data(unlabled_X, unlabled_X)
    return unlabled_X_train, unlabled_X_test, unlabled_Y_train, unlabled_Y_test


def using_labeled_data_for_autoencode(labeled_X):
    # df = read_data('../DMC_2019_task/train.csv')
    # X, y = labeled_data(df)
    # X = scale(X, scaler)
    X_train, X_test, Y_train, Y_test = split_data(labeled_X, labeled_X)
    return X_train, X_test, Y_train, Y_test


class ModelFactory:

    @staticmethod
    def createDecisionTree():
        return DecisionTreeClassifier()

    @staticmethod
    def createSVM():
        return svm.SVC(kernel='linear', C=0.5, degree=3)

    @staticmethod
    def createLogistic():
        return LogisticRegression()

    @staticmethod
    def createOneClassSVM():
        return svm.OneClassSVM(kernel='rbf', degree=3, gamma=0.01)


feature_cols = ['trustLevel_1', 'trustLevel_2', 'trustLevel_3', 'trustLevel_4', 'trustLevel_5', 'trustLevel_6',
                'totalScanTimeInSeconds', 'grandTotal', 'lineItemVoids', 'scansWithoutRegistration',
                'quantityModifications', 'scannedLineItemsPerSecond', 'valuePerSecond', 'lineItemVoidsPerPosition']


def decisionTreeClassifier(X_train, X_test, y_train, y_test):
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


def svmClassifier(X_train, X_test, y_train, y_test):
    SVM = ModelFactory.createSVM()
    SVM.fit(X_train, y_train)
    y_SVM_pred = SVM.predict(X_test)
    SVM_cm = metrics.confusion_matrix(y_test, y_SVM_pred)
    print("Confusion matrix using SVM", SVM_cm)
    print("SVM scores:", SVM_cm[1][1] * 5 - SVM_cm[0][1] * 25 - SVM_cm[1][0] * 5)


def oneClassSVM(X_train, y_train, X_test, y_test):
    ocSVM = ModelFactory.createOneClassSVM()
    ocSVM.fit(X_train)
    y_ocSVM_pred = ocSVM.predict(X_test)
    print(y_ocSVM_pred)
    # ocSVM_cm=metrics.confusion_matrix(normal_y_test,y_ocSVM_pred)
    # print("Confusion matrix using One class SVM", ocSVM_cm)


def logisticClassifier(X_train, X_test, y_train, y_test):
    lc = ModelFactory.creatLogistic()
    lc.fit(X_train, y_train)
    y_lc_pred = lc.predict(X_test)
    lc_cm = metrics.confusion_matrix(y_test, y_lc_pred)
    print("Confusion matrix using Logistic", lc_cm)
    print("Logistic scores:", lc_cm[1][1] * 5 - lc_cm[0][1] * 25 - lc_cm[1][0] * 5)


def votingClassifier(X_train, X_test, y_train, y_test):
    votingClassifier = VotingClassifier(estimators=[
        ('dc', ModelFactory.createDecisionTree()), ('svm', ModelFactory.createSVM()),
        ('lg', ModelFactory.createLogistic())
    ], voting='hard')
    votingClassifier.fit(X_train, y_train)
    y_voted = votingClassifier.predict(X_test)
    voting_cm = metrics.confusion_matrix(y_test, y_voted)
    print("Confusion matrix using Voting", voting_cm)
    print("Voting scores:", voting_cm[1][1] * 5 - voting_cm[0][1] * 25 - voting_cm[1][0] * 5)


def scores(y_test, y_pred):
    cf_matrix = K.variable(metrics.confusion_matrix(y_test, y_pred))
    print(cf_matrix.shape)
    return K.variable(np.array([[(cf_matrix[1][1] * 5 - cf_matrix[0][1] * 25 - cf_matrix[1][0] * 5) / len(y_test)]]))


def createDeepModel():
    model = Sequential()
    activation = 'tanh'
    leakyReLuLayer = tf.keras.layers.LeakyReLU(alpha=0.001)
    l = 1e-8
    units = [2]
    model.add(layers.Dense(units=12, input_dim=14, activation=activation, kernel_regularizer=regularizers.l1(l)))
    # model.add(leakyReLuLayer)
    model.add(layers.Dropout(DROP_OUT_PROB))
    for i in range(len(units)):
        model.add(layers.Dense(units=units[i], activation=activation, kernel_regularizer=regularizers.l1(l)))
        # model.add(layers.Dense(units=units[i],activation='relu', kernel_regularizer=regularizers.l2(l)))
        # model.add(leakyReLuLayer)
        model.add(layers.Dropout(DROP_OUT_PROB))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def deepNN(X, y):
    kFold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=50)
    for i, (train_index, val_index) in enumerate(kFold.split(X, y)):
        print("Running fold {} / {}".format(i + 1, n_folds))
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_train),
                                                          y_train)
        run_deep_nn(X_train, y_train, X_val, y_val, class_weights, i)


def run_deep_nn(X_train, y_train, X_val, y_val, class_weights, i_th_fold):
    custom_metrics = CustomMetrics()
    model = createDeepModel()
    model_cp = ModelCheckpoint(
        filepath='../models/deep-nn/fold_' + str(i_th_fold + 1) + '_deep_supervised_model.h5',
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                        validation_data=[X_val, y_val], class_weight=class_weights, verbose=0,
                        callbacks=[model_cp, model_es, custom_metrics]).history
    train_loss = history['loss']
    val_loss = history['val_loss']
    pyplot.plot(train_loss, label='train')
    pyplot.plot(val_loss, label='val')
    pyplot.title('Deep_nn_fold_'+str(i_th_fold))
    pyplot.legend()
    pyplot.show()
    for i in range(len(custom_metrics.avg_scores)):
        print(str(custom_metrics.confusion[i]) + '---->' + str(custom_metrics.avg_scores[i]))


def deep_autoencoder(X_train, X_test, Y_train, Y_test):
    model = Sequential()
    # leakyReLuLayer = tf.keras.layers.LeakyReLU(alpha=0.005)
    l = 1e-7
    units = []
    n_units = len(units)
    activation = 'tanh'
    # l = 0.001
    # encoder
    model.add(layers.Dense(units=12, input_dim=14, activation=activation, kernel_regularizer=regularizers.l2(l)))
    # model.add(leakyReLuLayer)
    # model.add(layers.Dropout(DROP_OUT_PROB))
    for i in range(n_units):
        model.add(layers.Dense(units=units[i], activation=activation, kernel_regularizer=regularizers.l2(l)))
        # model.add(leakyReLuLayer)
        # model.add(layers.Dropout(DROP_OUT_PROB))
        # model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=2, activation=activation, kernel_regularizer=regularizers.l2(l)))
    # decoder

    for i in range(n_units):
        model.add(layers.Dense(units=units[n_units - i - 1], activation=activation))
        # model.add(leakyReLuLayer)
        # model.add(layers.Dropout(DROP_OUT_PROB))
        # model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=12, activation=activation))
    # model.add(leakyReLuLayer)
    # model.add(layers.Dropout(DROP_OUT_PROB))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=14, activation=activation))
    # model.add(leakyReLuLayer)
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')
    model.summary()
    model_cp = ModelCheckpoint(filepath="../models/pre-trained/model_autoencoder.h5",
                               save_best_only=True,
                               save_weights_only=False, monitor='val_loss', mode='min',
                               verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
    tensorboard = TensorBoard(log_dir='/tmp/logs/autoencoder')
    BATCH_SIZE = 128
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        shuffle=True, verbose=1, validation_data=[X_test, Y_test],
                        callbacks=[model_es, model_cp, tensorboard]).history

    train_loss = history['loss']
    val_loss = history['val_loss']
    pyplot.plot(train_loss, label='train')
    pyplot.plot(val_loss, label='val')
    pyplot.legend()
    pyplot.show()


def plot_loss(train_loss, val_loss, epochs):
    # epochs = range(EPOCHS)
    plt.figure()
    plt.plot(epochs, train_loss, color='blue', label='Training loss')
    plt.plot(epochs, val_loss, color='red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def hybrid_model(X, y):
    kFold = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for i, (train_index, val_index) in enumerate(kFold.split(X, y)):
        print("Running fold {} / {}".format(i + 1, n_folds))
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_train),
                                                          y_train)
        print('---> Running Hybrids')
        run_hybrid_model(X_train, y_train, X_val, y_val, class_weights, i)


def run_hybrid_model(X_train, y_train, X_val, y_val, class_weights, i_th_fold):
    base_model = load_model('../models/pre-trained/model_autoencoder.h5')
    print("Loaded model from disk")
    base_model.trainable = True
    custom_metrics = CustomMetrics()
    model = createDeepModel()
    model.add(layers.Dense(units=1, activation='sigmoid', name='dense_sp'))
    model.layers[0].set_weights(base_model.layers[0].get_weights())
    model.layers[2].set_weights(base_model.layers[1].get_weights())
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model_cp = ModelCheckpoint(
        filepath='../models/hybrids/' + 'fold_' + str(i_th_fold + 1) + '_combined_autoencoder_supervised_model.h5',
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                        validation_data=[X_val, y_val], class_weight=class_weights,verbose=0,
                        callbacks=[model_cp, model_es, custom_metrics]).history
    train_loss = history['loss']
    val_loss = history['val_loss']
    pyplot.plot(train_loss, label='train')
    pyplot.plot(val_loss, label='val')
    pyplot.title('Hybrid_fold_'+str(i_th_fold))
    pyplot.legend()
    pyplot.show()
    for i in range(len(custom_metrics.avg_scores)):
        print(str(custom_metrics.confusion[i]) + '---->' + str(custom_metrics.avg_scores[i]))


def compare_deep_nn_with_hybrid(X, y):

    kFold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=50)
    for i, (train_index, val_index) in enumerate(kFold.split(X, y)):
        print("Running fold {} / {}".format(i + 1, n_folds))
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_train),
                                                          y_train)

        print('---> Running deep NN')
        run_deep_nn(X_train, y_train, X_val, y_val, class_weights, i)
        print('---> Running Hybrids')
        run_hybrid_model(X_train, y_train, X_val, y_val, class_weights, i)


def pre_trained(X):
    X_train, X_test, Y_train, Y_test = split_data(X, X)
    deep_autoencoder(X_train, X_test, Y_train, Y_test)


def supervised_learning(scaled_X, y):
    # decisionTreeClassifier(X_train, X_test, y_train, y_test)
    # svmClassifier()
    # logisticClassifier()
    # votingClassifier()
    # deepNN(scaled_X, y)
    # combine_deep_autoencoder_and_supervised_model(scaled_X,y)
    compare_deep_nn_with_hybrid(scaled_X,y)


if __name__ == '__main__':
    print('-----> Training........')
    labeled_df = read_data('../DMC_2019_task/train.csv')
    unlabeled_df = read_data('../DMC_2019_task/test.csv')
    labeled_X, labeled_y = labeled_data(labeled_df)
    unlabeled_X = one_hot_trust_level(unlabeled_df)
    X = pd.concat([labeled_X, unlabeled_X])
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    labeled_X = scale(labeled_X, scaler)
    unlabeled_X = scale(unlabeled_X, scaler)
    #pre_trained(X)
    supervised_learning(labeled_X, labeled_y)
