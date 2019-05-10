from matplotlib import pyplot
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras import Sequential, layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from custom_metrics import CustomMetrics
from preprocessing import *
from custom_losses import *
from evaluate import *

DROP_OUT_PROB = 0.05
EPOCHS = 5000
BATCH_SIZE = 32
n_folds = 3
PATIENCE = 300


#
# def load_unlabeled_data(scaler):
#     unlabled_X = scale(one_hot_trust_level(read_data('../DMC_2019_task/test.csv')), scaler)
#     unlabled_X_train, unlabled_X_test, unlabled_Y_train, unlabled_Y_test = split_data(unlabled_X, unlabled_X)
#     return unlabled_X_train, unlabled_X_test, unlabled_Y_train, unlabled_Y_test
#
#
# def using_labeled_data_for_autoencode(labeled_X):
#     # df = read_data('../DMC_2019_task/train.csv')
#     # X, y = labeled_data(df)
#     # X = scale(X, scaler)
#     X_train, X_test, Y_train, Y_test = split_data(labeled_X, labeled_X)
#     return X_train, X_test, Y_train, Y_test


def scores(y_test, y_pred):
    cf_matrix = K.variable(metrics.confusion_matrix(y_test, y_pred))
    print(cf_matrix.shape)
    return K.variable(np.array([[(cf_matrix[1][1] * 5 - cf_matrix[0][1] * 25 - cf_matrix[1][0] * 5) / len(y_test)]]))


def createDeepModel():
    model = Sequential()
    relu = 'relu'
    leakyReLuLayer = tf.keras.layers.LeakyReLU(alpha=0.1)
    l = 1e-7
    units = [4]
    model.add(layers.Dense(units=6, input_dim=14, activation=relu, kernel_regularizer=regularizers.l2(l)))
    # model.add(leakyReLuLayer)
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROP_OUT_PROB))
    for i in range(len(units)):
        model.add(layers.Dense(units=units[i], activation=relu, kernel_regularizer=regularizers.l2(l)))
        # model.add(leakyReLuLayer)
        # model.add(layers.BatchNormalization())
        model.add(layers.Dropout(DROP_OUT_PROB))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    # adam=Adam(lr=0.001,decay=0.0005)
    # model.compile(optimizer=adam, loss=K.binary_crossentropy,metrics=['crossentropy'])
    return model


def compute_class_weight(y_train):
    # unique_labels = []
    # for i in range(len(y_train)):
    #     if y_train[i][0] == 1:
    #         unique_labels.append(0)
    #     else:
    #         unique_labels.append(1)
    # unique_labels = np.array(unique_labels)
    cw = class_weight.compute_class_weight('balanced', np.unique(y_train),
                                           y_train)
    return cw


def deep_nn(X_train, y_train, X_test, y_test):
    kFold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=500)
    deep_nn_models = dict()
    for i, (train_index, val_index) in enumerate(kFold.split(X_train, y_train)):
        print("Running fold {} / {}".format(i + 1, n_folds))
        X_train_fold, y_train_fold = X_train[train_index], y_train.iloc[train_index]
        X_val, y_val = X_train[val_index], y_train.iloc[val_index]
        cw = compute_class_weight(y_train)
        model = run_deep_nn_fold(X_train_fold, y_train_fold, X_val, y_val, cw, i)
        deep_nn_models[i] = model
    print('****Evaluatating deep NN****')
    for i, model in deep_nn_models.items():
        print('----> Fold_' + str(i))
        evaluate(model, X_test, y_test)


def run_deep_nn_fold(X_train_fold, y_train_fold, X_val, y_val, class_weights, i_th_fold):
    print(class_weights)
    custom_metrics = CustomMetrics()
    model = createDeepModel()
    model.compile(optimizer='adam', loss=custom_focal_loss)
    model_cp = ModelCheckpoint(
        filepath='../models/deep-nn/fold_' + str(i_th_fold + 1) + '_deep_supervised_model.h5',
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=PATIENCE)
    history = model.fit(X_train_fold, y_train_fold, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                        validation_data=[X_val, y_val], verbose=0, class_weight=None,
                        callbacks=[model_cp, model_es, custom_metrics]).history
    plot_history(history, 'Deep_nn_fold_', i_th_fold)
    print_metrics(custom_metrics)
    return model


def deep_hybrid_model(X_train, y_train, X_test, y_test):
    kFold = KFold(n_splits=n_folds, shuffle=True)
    hybrid_models = dict()
    for i, (train_index, val_index) in enumerate(kFold.split(X_train, y_train)):
        print("Running fold {} / {}".format(i + 1, n_folds))
        X_train_fold, y_train_fold = X_train[train_index], y_train.iloc[train_index]
        X_val, y_val = X_train[val_index], y_train.iloc[val_index]
        cw = compute_class_weight(y_train)
        print('---> Running Hybrids')
        hb_model = run_hybrid_model(X_train_fold, y_train_fold, X_val, y_val, cw, i)
        hybrid_models[i] = hb_model
    print('****Evaluating hybrid****')
    for i, model in hybrid_models.items():
        print('----> Fold_' + str(i))
        evaluate(model, X_test, y_test)


def run_hybrid_model(X_train_fold, y_train_fold, X_val, y_val, class_weights, i_th_fold):
    base_model = load_model('../models/pre-trained/model_autoencoder_relu_6_4_lr_0001.h5')
    print("Loaded model from disk")
    base_model.trainable = True
    custom_metrics = CustomMetrics()
    model = createDeepModel()
    model.add(layers.Dense(units=1, activation='sigmoid', name='dense_sp'))
    model.layers[0].set_weights(base_model.layers[0].get_weights())
    model.layers[2].set_weights(base_model.layers[2].get_weights())
    # model.layers[4].set_weights(base_model.layers[4].get_weights())
    model.compile(optimizer='adam', loss=custom_focal_loss)
    model_cp = ModelCheckpoint(
        filepath='../models/hybrids/' + 'fold_' + str(i_th_fold + 1) + '_combined_autoencoder_supervised_model.h5',
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=PATIENCE)
    history = model.fit(X_train_fold, y_train_fold, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                        validation_data=[X_val, y_val], verbose=0, class_weight=class_weights,
                        callbacks=[model_cp, model_es, custom_metrics]).history

    plot_history(history, 'Hybrid_fold_', i_th_fold)
    print_metrics(custom_metrics)
    return model


def run_transfer_model(X_train, y_train, X_val, y_val, class_weights, i_th_fold):
    base_model = load_model('../models/pre-trained/model_autoencoder_relu_6_4_lr_0001.h5')
    print("Loaded model from disk")
    base_model.trainable = True
    custom_metrics = CustomMetrics()
    model = createDeepModel()
    model.add(layers.Dense(units=1, activation='sigmoid', name='dense_sp'))
    model.layers[0].set_weights(base_model.layers[0].get_weights())
    model.layers[2].set_weights(base_model.layers[2].get_weights())
    # model.layers[4].set_weights(base_model.layers[4].get_weights())
    model.compile(optimizer='adam', loss=custom_focal_loss)
    model_cp = ModelCheckpoint(
        filepath='../models/hybrids/' + 'fold_' + str(i_th_fold + 1) + '_transfer_supervised_model.h5',
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=300)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                        validation_data=[X_val, y_val], verbose=0, class_weight=class_weights,
                        callbacks=[model_cp, model_es, custom_metrics]).history
    train_loss = history['loss']
    val_loss = history['val_loss']
    pyplot.plot(train_loss, label='train')
    pyplot.plot(val_loss, label='val')
    pyplot.title('Hybrid_fold_' + str(i_th_fold))
    pyplot.legend()
    pyplot.show()
    for i in range(len(custom_metrics.avg_scores)):
        print(str(custom_metrics.confusion[i]) + '---->' + str(custom_metrics.avg_scores[i]))


def compare_deep_nn_with_hybrid(X_train, y_train, X_test, y_test):
    kFold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1000)
    deep_nn_models = dict()
    hybrid_models = dict()
    for i, (train_index, val_index) in enumerate(kFold.split(X_train, y_train)):
        print("Running fold {} / {}".format(i + 1, n_folds))
        X_train_fold, y_train_fold = X_train[train_index], y_train[train_index]
        X_val, y_val = X_train[val_index], y_train[val_index]
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_train),
                                                          y_train)

        print('---> Running deep NN')
        deep_nn_model = run_deep_nn_fold(X_train_fold, y_train_fold, X_val, y_val, X_test, y_test, class_weights, i)
        print('---> Running Hybrids')
        hb_model = run_hybrid_model(X_train_fold, y_train_fold, X_val, y_val, X_test, y_test, class_weights, i)
        deep_nn_models[i] = deep_nn_model
        hybrid_models[i] = hb_model

    print('****Evaluatating deep NN****')
    for i, model in deep_nn_models.items():
        print('----> Fold_' + str(i))
        evaluate(model, X_test, y_test)
    print('\n')
    print('****Evaluating hybrid****')
    for i, model in hybrid_models.items():
        print('----> Fold_' + str(i))
        evaluate(model, X_test, y_test)


def plot_history(history, title, i_th_fold):
    train_loss = history['loss']
    val_loss = history['val_loss']
    pyplot.plot(train_loss, label='train')
    pyplot.plot(val_loss, label='val')
    pyplot.title(title + str(i_th_fold))
    pyplot.legend()
    pyplot.show()


def print_metrics(custom_metrics):
    for i in range(len(custom_metrics.avg_scores)):
        print(str(custom_metrics.confusion[i]) + '---->' + str(custom_metrics.avg_scores[i]))


def training(X_train, y_train, X_test, y_test):
    #deep_nn(X_train, y_train, X_test, y_test)
    deep_hybrid_model(X_train, y_train, X_test, y_test)
    # compare_deep_nn_with_hybrid(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    print('-----> Training........')
    X_train,X_test,y_train,y_test=prepare_data(StandardScaler())
    training(X_train, y_train, X_test, y_test)
