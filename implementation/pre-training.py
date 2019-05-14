import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import *

from preprocessing import *

# from custom_binary_crossentropy import *

DROP_OUT_PROB = 0.05
EPOCHS = 5000
BATCH_SIZE = 32
n_folds = 5
ACTIVATION='tanh'

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


feature_cols = ['trustLevel_1', 'trustLevel_2', 'trustLevel_3', 'trustLevel_4', 'trustLevel_5', 'trustLevel_6',
                'totalScanTimeInSeconds', 'grandTotal', 'lineItemVoids', 'scansWithoutRegistration',
                'quantityModifications', 'scannedLineItemsPerSecond', 'valuePerSecond', 'lineItemVoidsPerPosition']


def deep_autoencoder(X_train, X_test, Y_train, Y_test):
    model = Sequential()
    leakyReLuLayer = tf.keras.layers.LeakyReLU()
    l = 1e-7
    units = []
    n_units = len(units)
    # encoder
    model.add(layers.Dense(units=6, input_dim=14,activation=ACTIVATION, kernel_regularizer=regularizers.l2(l)))
    #model.add(leakyReLuLayer)
    model.add(layers.Dropout(DROP_OUT_PROB))
    for i in range(n_units):
        model.add(layers.Dense(units=units[i], activation=ACTIVATION,kernel_regularizer=regularizers.l2(l)))
        #model.add(leakyReLuLayer)
        model.add(layers.Dropout(DROP_OUT_PROB))
        #model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=4,activation=ACTIVATION, kernel_regularizer=regularizers.l2(l)))
    #model.add(leakyReLuLayer)
    model.add(layers.Dropout(DROP_OUT_PROB))
    # decoder

    for i in range(n_units):
        model.add(layers.Dense(units=units[n_units - i - 1],activation=ACTIVATION))
        #model.add(leakyReLuLayer)
        model.add(layers.Dropout(DROP_OUT_PROB))
        #model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=6,activation=ACTIVATION))
    #model.add(leakyReLuLayer)
    model.add(layers.Dropout(DROP_OUT_PROB))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=14))
    #model.add(leakyReLuLayer)
    adam = Adam(lr=0.0001,decay=0.0001/EPOCHS)
    model.compile(optimizer=adam, loss='mse')
    model.summary()
    model_cp = ModelCheckpoint(
        filepath="../models/pre-trained/model_autoencoder_relu_6_4_lr_00001_tanh.h5",
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)
    tensorboard = TensorBoard(log_dir='/tmp/logs/autoencoder')
    BATCH_SIZE = 256
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        shuffle=True, verbose=0, validation_data=[X_test, Y_test],
                        callbacks=[model_es, model_cp, tensorboard]).history

    train_loss = history['loss']
    val_loss = history['val_loss']
    pyplot.plot(train_loss, label='train')
    pyplot.plot(val_loss, label='val')
    pyplot.legend()
    pyplot.show()

# def deep_autoencoder(X_train, X_test, Y_train, Y_test):
#     model = Sequential()
#     leakyReLuLayer = tf.keras.layers.LeakyReLU()
#     l = 1e-2
#     units = []
#     n_units = len(units)
#     # activation = 'relu'
#     # l = 0.001
#     # encoder
#     model.add(layers.Dense(units=8, input_dim=14, kernel_regularizer=regularizers.l2(l)))
#     model.add(leakyReLuLayer)
#     model.add(layers.Dropout(DROP_OUT_PROB))
#     for i in range(n_units):
#         model.add(layers.Dense(units=units[i], kernel_regularizer=regularizers.l2(l)))
#         model.add(leakyReLuLayer)
#         model.add(layers.Dropout(DROP_OUT_PROB))
#         #model.add(layers.BatchNormalization())
#     model.add(layers.Dense(units=4, kernel_regularizer=regularizers.l2(l)))
#     model.add(leakyReLuLayer)
#     model.add(layers.Dropout(DROP_OUT_PROB))
#     # decoder
#
#     for i in range(n_units):
#         model.add(layers.Dense(units=units[n_units - i - 1]))
#         model.add(leakyReLuLayer)
#         model.add(layers.Dropout(DROP_OUT_PROB))
#         #model.add(layers.BatchNormalization())
#     model.add(layers.Dense(units=8))
#     model.add(leakyReLuLayer)
#     model.add(layers.Dropout(DROP_OUT_PROB))
#     #model.add(layers.BatchNormalization())
#     model.add(layers.Dense(units=14))
#     model.add(leakyReLuLayer)
#     adam = Adam(lr=0.001)
#     model.compile(optimizer=adam, loss='mse')
#     model.summary()
#     model_cp = ModelCheckpoint(
#         filepath="../models/pre-trained/model_autoencoder_relu_8_4_lr_001_leakyrelu_batchnorm.h5",
#         save_best_only=True,
#         save_weights_only=False, monitor='val_loss', mode='min',
#         verbose=1)
#     model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
#     tensorboard = TensorBoard(log_dir='/tmp/logs/autoencoder')
#     BATCH_SIZE = 256
#     history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
#                         shuffle=True, verbose=0, validation_data=[X_test, Y_test],
#                         callbacks=[model_es, model_cp, tensorboard]).history
#
#     train_loss = history['loss']
#     val_loss = history['val_loss']
#     pyplot.plot(train_loss, label='train')
#     pyplot.plot(val_loss, label='val')
#     pyplot.legend()
#     pyplot.show()

def plot_loss(train_loss, val_loss, epochs):
    # epochs = range(EPOCHS)
    plt.figure()
    plt.plot(epochs, train_loss, color='blue', label='Training loss')
    plt.plot(epochs, val_loss, color='red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def pre_trained(X):
    X_train, X_test, Y_train, Y_test = split_data(X, X)
    deep_autoencoder(X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
    print('-----> Training........')
    labeled_df = read_data('../DMC_2019_task/train.csv')
    unlabeled_df = read_data('../DMC_2019_task/test.csv')
    X_train, X_test, y_train, y_test = labeled_data(labeled_df)
    unlabeled_X = one_hot_trust_level(unlabeled_df)
    X = pd.concat([X_train,X_test, unlabeled_X])
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    unlabeled_X = scale(unlabeled_X, scaler)
    pre_trained(unlabeled_X)
