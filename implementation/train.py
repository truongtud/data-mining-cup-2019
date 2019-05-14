from keras.utils import plot_model
from matplotlib import pyplot
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import class_weight
from tensorflow.keras import Sequential, layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *

from custom_losses import *
from custom_metrics import CustomMetrics
from evaluate import *
from preprocessing import *

DROP_OUT_PROB = 0.05
EPOCHS = 6000
BATCH_SIZE = 32
n_folds = 8
PATIENCE = 300
ACTIVATION = 'tanh'
DECODER = 'decoder'
CLASSIFIER = 'classifier'
CLASSIFIER_LOSS = custom_focal_loss
isleakyrelu = False
K.binary_crossentropy

def scores(y_test, y_pred):
    cf_matrix = K.variable(metrics.confusion_matrix(y_test, y_pred))
    print(cf_matrix.shape)
    return K.variable(np.array([[(cf_matrix[1][1] * 5 - cf_matrix[0][1] * 25 - cf_matrix[1][0] * 5) / len(y_test)]]))


def createDeepModel():
    model = Sequential()
    leakyReLuLayer = tf.keras.layers.LeakyReLU(alpha=0.1)
    l = 1e-7
    units = [4]
    if not isleakyrelu:
        model.add(layers.Dense(units=6, input_dim=14, activation=ACTIVATION, kernel_regularizer=regularizers.l2(l)))
    else:
        model.add(layers.Dense(units=6, input_dim=14, kernel_regularizer=regularizers.l2(l)))
        model.add(leakyReLuLayer)
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROP_OUT_PROB))
    for i in range(len(units)):
        if not isleakyrelu:
            model.add(layers.Dense(units=units[i], activation=ACTIVATION, kernel_regularizer=regularizers.l2(l)))
        else:
            model.add(layers.Dense(units=units[i], kernel_regularizer=regularizers.l2(l)))
            model.add(leakyReLuLayer)
        # model.add(layers.BatchNormalization())
        model.add(layers.Dropout(DROP_OUT_PROB))
    model.add(layers.Dense(units=1, activation='sigmoid'))
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
    init_lr = 0.0001
    adam = Adam(lr=init_lr, decay=init_lr / EPOCHS)
    model.compile(optimizer=adam, loss=CLASSIFIER_LOSS)
    model_cp = ModelCheckpoint(
        filepath='../models/deep-nn/fold_deep_supervised_model_6_4_lr0001_' + str(
            CLASSIFIER_LOSS.__name__) + '_' + str(ACTIVATION) + '_' + str(i_th_fold) + '.h5',
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=PATIENCE)
    history = model.fit(X_train_fold, y_train_fold, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                        validation_data=(X_val, y_val), verbose=0, class_weight=class_weights,
                        callbacks=[model_cp, model_es, custom_metrics]).history
    plot_history(history, 'Deep_nn_fold_', i_th_fold)
    print_metrics(custom_metrics)
    return model


def deep_hybrid_model(X_train, y_train, X_test, y_test):
    kFold = KFold(n_splits=n_folds, shuffle=True, random_state=500)
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
    # model.add(layers.Dense(units=1, activation='sigmoid', name='dense_sp'))
    model.layers[0].set_weights(base_model.layers[0].get_weights())
    model.layers[2].set_weights(base_model.layers[2].get_weights())
    # model.layers[4].set_weights(base_model.layers[4].get_weights())
    init_lr = 0.0001
    adam = Adam(lr=init_lr, decay=init_lr / EPOCHS)
    model.compile(optimizer=adam, loss=CLASSIFIER_LOSS)
    model_cp = ModelCheckpoint(
        filepath='../models/hybrids/fold_combined_supervised_autoencoder_model_6_4_lr0001_' + str(
            CLASSIFIER_LOSS.__name__) + '_' + str(ACTIVATION) + '_' + str(i_th_fold) + '.h5',
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=PATIENCE)
    history = model.fit(X_train_fold, y_train_fold, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                        validation_data=(X_val, y_val), verbose=0, class_weight=class_weights,
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
    model.compile(optimizer='adam', loss=CLASSIFIER_LOSS)
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


def deep_supervised_autoencoder(X_train, y_train, X_test, y_test):
    kFold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=500)
    deep_supervised_autoencoder_models = dict()
    for i, (train_index, val_index) in enumerate(kFold.split(X_train, y_train)):
        print("Running fold {} / {}".format(i + 1, n_folds))
        X_train_fold, y_train_fold = X_train[train_index], y_train.iloc[train_index]
        X_val, y_val = X_train[val_index], y_train.iloc[val_index]
        cw = compute_class_weight(y_train)
        model = run_deep_supervised_autoencoder_fold(X_train_fold, y_train_fold, X_val, y_val, cw, i,
                                                     use_pre_trained=True)
        deep_supervised_autoencoder_models[i] = model
    print('****Evaluatating deep supervised autoencoder****')
    for i, model in deep_supervised_autoencoder_models.items():
        print('----> Fold_' + str(i))
        evaluate(model, X_test, y_test, 'deep_supervised_autoencoder')


def build_deep_supervised_autoencoder_model():
    leakyReLuLayer = tf.keras.layers.LeakyReLU()
    l = 1e-7
    units = []
    n_units = len(units)
    encoder = None
    input = layers.Input(shape=(14,))
    if not isleakyrelu:
        encoder = layers.Dense(units=6, activation=ACTIVATION, kernel_regularizer=regularizers.l2(l))(input)
    else:
        encoder = layers.Dense(units=6, kernel_regularizer=regularizers.l2(l))(input)
        encoder = leakyReLuLayer(encoder)
    encoder = layers.Dropout(DROP_OUT_PROB)(encoder)
    # encoder=layers.BatchNormalization(encoder)
    for i in range(n_units):
        if not isleakyrelu:
            encoder = layers.Dense(units=units[i], activation=ACTIVATION, kernel_regularizer=regularizers.l2(l))(
                encoder)
        else:
            encoder = layers.Dense(units=units[i], kernel_regularizer=regularizers.l2(l))(encoder)
            encoder = leakyReLuLayer(encoder)
        encoder = layers.Dropout(DROP_OUT_PROB)(encoder)
        # encoder=layers.BatchNormalization(encoder)
    if not isleakyrelu:
        encoder = layers.Dense(units=4, activation=ACTIVATION, kernel_regularizer=regularizers.l2(l))(encoder)
    else:
        encoder = layers.Dense(units=4, kernel_regularizer=regularizers.l2(l))(encoder)
        encoder = leakyReLuLayer(encoder)
    latent = layers.Dropout(DROP_OUT_PROB)(encoder)
    # latent=layers.BatchNormalization(latent)
    # decoder
    decoder = latent
    for i in range(n_units):
        if not isleakyrelu:
            decoder = layers.Dense(units=units[n_units - i - 1], activation=ACTIVATION,
                                   kernel_regularizer=regularizers.l2(l))(decoder)
        else:
            decoder = layers.Dense(units=units[n_units - i - 1],
                                   kernel_regularizer=regularizers.l2(l))(decoder)
            decoder = leakyReLuLayer(decoder)
        decoder = layers.Dropout(DROP_OUT_PROB)(decoder)
        # decoder=layers.BatchNormalization(decoder)
        # model.add(layers.BatchNormalization())
    if not isleakyrelu:
        decoder = layers.Dense(units=6, activation=ACTIVATION, kernel_regularizer=regularizers.l2(l))(decoder)
    else:
        decoder = layers.Dense(units=6, kernel_regularizer=regularizers.l2(l))(decoder)
        decoder = leakyReLuLayer(decoder)
    decoder = layers.Dropout(DROP_OUT_PROB)(decoder)
    # decoder=layers.BatchNormalization(decoder)
    decoder = layers.Dense(units=14, name=DECODER)(decoder)
    classifier = layers.Dense(units=3, activation=ACTIVATION)(latent)
    classifier = layers.Dropout(DROP_OUT_PROB)(classifier)
    classifier = layers.Dense(units=1, activation='sigmoid', name=CLASSIFIER)(classifier)
    model = tf.keras.models.Model(inputs=input, outputs=[decoder, classifier], name='deep_supervised_autoencoder')
    losses = {
        DECODER: 'mse',
        CLASSIFIER: custom_focal_loss
    }
    loss_weights = {
        DECODER: 0.2,
        CLASSIFIER: 1.0
    }
    init_lr = 0.0001
    adam = Adam(lr=init_lr, decay=init_lr / EPOCHS)
    sgd=SGD(lr=init_lr,momentum=0.9,decay=init_lr/EPOCHS)
    model.compile(optimizer=adam, loss=losses, loss_weights=loss_weights)
    model.summary()
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    return model


def build_deep_supervised_autoencoder_model_batchnorm():
    leakyReLuLayer = tf.keras.layers.LeakyReLU()
    encoder = Sequential()
    l = 1e-7
    units = []
    n_units = len(units)
    # input = layers.Input(shape=(14,))
    # encoder.add(input)
    if not isleakyrelu:
        encoder.add(layers.Dense(units=6, activation=ACTIVATION, input_dim=14, kernel_regularizer=regularizers.l2(l)))
    else:
        encoder.add(layers.Dense(units=6, input_dim=14, kernel_regularizer=regularizers.l2(l)))
        encoder.add(leakyReLuLayer())
    encoder.add(layers.Dropout(DROP_OUT_PROB))
    encoder.add(layers.BatchNormalization())
    for i in range(n_units):
        if not isleakyrelu:
            encoder.add(layers.Dense(units=units[i], activation=ACTIVATION, kernel_regularizer=regularizers.l2(l)))
        else:
            encoder.add(layers.Dense(units=units[i], kernel_regularizer=regularizers.l2(l)))
            encoder.add(leakyReLuLayer())
        encoder.add(layers.Dropout(DROP_OUT_PROB)())
        encoder.add(layers.BatchNormalization())
    if not isleakyrelu:
        encoder.add(layers.Dense(units=4, activation=ACTIVATION, kernel_regularizer=regularizers.l2(l)))
    else:
        encoder.add(layers.Dense(units=4, kernel_regularizer=regularizers.l2(l)))
        encoder.add(leakyReLuLayer())
    encoder.add(layers.Dropout(DROP_OUT_PROB))
    encoder.add(layers.BatchNormalization(name='latent'))
    # decoder
    decoder = Sequential()
    decoder.add(encoder.get_layer('latent'))
    for i in range(n_units):
        if not isleakyrelu:
            decoder.add(layers.Dense(units=units[n_units - i - 1], activation=ACTIVATION,
                                     kernel_regularizer=regularizers.l2(l)))
        else:
            decoder.add(layers.Dense(units=units[n_units - i - 1],
                                     kernel_regularizer=regularizers.l2(l)))
            decoder.add(leakyReLuLayer())
        decoder.add(layers.Dropout(DROP_OUT_PROB))
        decoder.add(layers.BatchNormalization())
    if not isleakyrelu:
        decoder.add(layers.Dense(units=6, activation=ACTIVATION, kernel_regularizer=regularizers.l2(l)))
    else:
        decoder.add(layers.Dense(units=6, kernel_regularizer=regularizers.l2(l)))
        decoder.add(leakyReLuLayer())
    decoder.add(layers.Dropout(DROP_OUT_PROB))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Dense(units=14, name=DECODER))
    classifier = Sequential()
    classifier.add(encoder.get_layer('latent'))
    classifier.add(layers.Dense(units=1, activation='sigmoid', name=CLASSIFIER))
    model = tf.keras.models.Model(inputs=encoder.inputs, outputs=[decoder.output, classifier.output],
                                  name='deep_supervised_autoencoder')
    losses = {
        DECODER: 'mse',
        CLASSIFIER: custom_focal_loss
    }
    loss_weights = {
        DECODER: 0.2,
        CLASSIFIER: 1.0
    }
    init_lr = 0.0001
    adam = Adam(lr=init_lr, decay=init_lr / EPOCHS)
    model.compile(optimizer=adam, loss=losses, loss_weights=loss_weights)
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    model.summary()
    return model


def run_deep_supervised_autoencoder_fold(X_train_fold, y_train_fold, X_val, y_val, class_weights, i_th_fold,
                                         use_pre_trained=False):
    base_model = load_model('../models/pre-trained/model_autoencoder_relu_6_4_lr_00001_tanh.h5')
    print("Loaded pre-trained model from disk")
    base_model.trainable = True
    custom_metrics = CustomMetrics('deep_supervised_autoencoder')
    model = build_deep_supervised_autoencoder_model()
    # model=build_deep_supervised_autoencoder_model_batchnorm()
    if use_pre_trained:
        model.layers[1].set_weights(base_model.layers[0].get_weights())
        if isleakyrelu:
            model.layers[5].set_weights(base_model.layers[2].get_weights())
        else:
            model.layers[3].set_weights(base_model.layers[2].get_weights())
    file_path = None
    if use_pre_trained:
        file_path = '../models/deep-supervised-autoencoder-using-pre-trained-tanh-without-class_weights/deep_supervised_autoencoder_with_fold_6_4_lr_00001_' + str(
            CLASSIFIER_LOSS.__name__) + '_' + str(ACTIVATION) + '_' + str(i_th_fold) + '.h5'
    else:
        file_path = '../models/deep-supervised-autoencoder/deep_supervised_autoencoder_with_fold_6_4_lr_00001_' + str(
            CLASSIFIER_LOSS.__name__) + '_' + str(ACTIVATION) + '_' + str(i_th_fold) + '.h5'
    model_cp = ModelCheckpoint(
        filepath=file_path,
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
    tensorboard = TensorBoard(log_dir='/tmp/logs/deep-supervised-autoencoder')
    history = model.fit(X_train_fold, {DECODER: X_train_fold, CLASSIFIER: y_train_fold}, batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        shuffle=True, verbose=0, validation_data=(X_val, {CLASSIFIER: y_val, DECODER: X_val}),
                        #class_weight={CLASSIFIER: class_weights},
                        callbacks=[model_es, model_cp, tensorboard, custom_metrics]).history

    plot_history(history, 'Deep_supervised_autoencoder_', i_th_fold)
    print_metrics(custom_metrics)
    return model


def run_deep_supervised_autoencoder_without_fold(X_train, y_train, X_test, y_test, use_pre_trained=False):
    base_model = load_model('../models/pre-trained/model_autoencoder_relu_6_4_lr_0001.h5')
    print("Loaded pre-trained model from disk")
    base_model.trainable = True
    model = build_deep_supervised_autoencoder_model()
    if use_pre_trained:
        model.layers[1].set_weights(base_model.layers[0].get_weights())
        if isleakyrelu:
            model.layers[4].set_weights(base_model.layers[2].get_weights())
        else:
            model.layers[3].set_weights(base_model.layers[2].get_weights())

    custom_metrics = CustomMetrics('deep_supervised_autoencoder')
    model_cp = ModelCheckpoint(
        filepath='../models/deep-supervised-autoencoder/deep_supervised_autoencoder_without_fold_6_4_lr_001_' + str(
            CLASSIFIER_LOSS.__name__) + '_' + str(ACTIVATION) + '.h5',
        save_best_only=True,
        save_weights_only=False, monitor='val_loss', mode='min',
        verbose=1)
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
    tensorboard = TensorBoard(log_dir='/tmp/logs/deep-supervised-autoencoder')
    history = model.fit(X_train, {DECODER: X_train, CLASSIFIER: y_train}, batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        shuffle=True, verbose=1, callbacks=[model_cp, model_es, tensorboard, custom_metrics],
                        validation_split=0.2).history
    plot_history(history, 'Deep_supervised_autoencoder_', 'without_fold')
    print_metrics(custom_metrics)
    evaluate(model, X_test, y_test, 'deep_supervised_autoencoder')


def compare_all(X_train, y_train, X_test, y_test):
    kFold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1000)
    deep_nn_models = dict()
    hybrid_models = dict()
    deep_supervised_autoencoder_models = dict()
    for i, (train_index, val_index) in enumerate(kFold.split(X_train, y_train)):
        print("Running fold {} / {}".format(i + 1, n_folds))
        X_train_fold, y_train_fold = X_train[train_index], y_train.iloc[train_index]
        X_val, y_val = X_train[val_index], y_train.iloc[val_index]
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_train),
                                                          y_train)

        print('---> Running deep NN')
        deep_nn_model = run_deep_nn_fold(X_train_fold, y_train_fold, X_val, y_val, class_weights, i)
        deep_nn_models[i] = deep_nn_model

        print('---> Running Hybrids')
        hb_model = run_hybrid_model(X_train_fold, y_train_fold, X_val, y_val, class_weights, i)
        hybrid_models[i] = hb_model

        print('---> Running Deep supervised autoencoder')
        dsa_model = run_deep_supervised_autoencoder_fold(X_train_fold, y_train_fold, X_val, y_val, class_weights, i)
        deep_supervised_autoencoder_models[i] = dsa_model

    print('****Evaluatating deep NN****')
    for i, model in deep_nn_models.items():
        print('----> Fold_' + str(i))
        evaluate(model, X_test, y_test)
    print('\n')
    print('****Evaluating hybrid****')
    for i, model in hybrid_models.items():
        print('----> Fold_' + str(i))
        evaluate(model, X_test, y_test)

    print('\n')
    print('****Evaluating deep supervised autoencoder****')
    for i, model in deep_supervised_autoencoder_models.items():
        print('----> Fold_' + str(i))
        evaluate(model, X_test, y_test, 'deep_supervised_autoencoder')


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
    # run_deep_supervised_autoencoder_without_fold(X_train, y_train, X_test, y_test,use_pre_trained=True)
    deep_supervised_autoencoder(X_train, y_train, X_test, y_test)
    # deep_nn(X_train, y_train, X_test, y_test)
    # deep_hybrid_model(X_train, y_train, X_test, y_test)
    # compare_all(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    print('-----> Training........')
    X_train, X_test, y_train, y_test = prepare_training_data(StandardScaler())
    training(X_train, y_train, X_test, y_test)
