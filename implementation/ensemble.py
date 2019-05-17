from keras.layers import Dense, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from custom_losses import custom_focal_loss
from preprocessing import *

'''
Follows this tutorial @link https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
'''
CLASSIFIER = 'classifier'
EPOCHS = 1000
DROP_OUT_PROB = 0.2


def load_models(path, n_models):
    models = []
    for i in range(n_models):
        file_path = path + str(i) + '.h5'
        model = load_model(file_path, custom_objects={'custom_focal_loss': custom_focal_loss})
        models.append(model)
    return models


def stacked_model(members, is_using_deep_supervised_autoencoder=False):
    ensemble_inputs = [model.input for model in members]
    ensemble_outputs = list()
    if is_using_deep_supervised_autoencoder:
        ensemble_outputs = [model.get_layer(CLASSIFIER).output for model in members]
    else:
        ensemble_outputs = [model.output for model in members]
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = False
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name

    merged_ensemble_outputs = concatenate(ensemble_outputs)
    #hidden_1 = Dense(10, activation='relu')(merged_ensemble_outputs)
    #hidden_1 = Dropout(DROP_OUT_PROB)(merged_ensemble_outputs)
    hidden_2 = Dense(6, activation='relu')(merged_ensemble_outputs)
    hidden_2 = Dropout(DROP_OUT_PROB)(hidden_2)
    hidden_3 = Dense(4, activation='relu')(hidden_2)
    hidden_3 = Dropout(DROP_OUT_PROB)(hidden_3)
    output = Dense(1, activation='sigmoid')(hidden_3)
    model = Model(inputs=ensemble_inputs, outputs=output)
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    init_lr = 0.0001
    adam = Adam(lr=init_lr, decay=init_lr / EPOCHS)
    model.compile(loss=custom_focal_loss, optimizer=adam, metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, X_train, y_train):
    # custom_metrics=CustomMetrics()
    X_trains = [X_train for _ in range(len(model.input))]
    cw = compute_class_weight(y_train)
    model.fit(X_trains, y_train, batch_size=32, epochs=EPOCHS, verbose=0, class_weight=cw)



def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


def print_metrics(custom_metrics):
    for i in range(len(custom_metrics.avg_scores)):
        print(str(custom_metrics.confusion[i]) + '---->' + str(custom_metrics.avg_scores[i]))


def compute_class_weight(y_train):
    cw = class_weight.compute_class_weight('balanced', np.unique(y_train),
                                           y_train)
    return cw


sub_models = load_models(
    '../models/deep-supervised-autoencoder-using-pre-trained-tanh-adam-gaussiannoise-dropout-005-gamma2-minmax_scale/deep_supervised_autoencoder_with_fold_6_4_lr_00001_custom_focal_loss_tanh_',
    2)
X_train, X_test, y_train, y_test = prepare_training_data(StandardScaler())
#def run_stacked_model(sub_models,X_train, X_test, y_train, y_test):
#s_model = stacked_model(sub_models, is_using_deep_supervised_autoencoder=True)
#fit_stacked_model(s_model, X_train, y_train)
#predicted_y = np.rint(predict_stacked_model(s_model, X_test))
#cf_m = confusion_matrix(y_test, predicted_y)
#print(cf_m.ravel())
print('sub models')
is_using_deep_supervised_autoencoder = True
for smodel in sub_models:
    predicted = None
    if is_using_deep_supervised_autoencoder:
        predicted = np.rint(smodel.predict(X_test)[1])
    else:
        predicted = np.rint(smodel.predict(X_test))
    cf = confusion_matrix(y_test, predicted)
    print(cf.ravel())



#run_stacked_model(sub_models,X_train, X_test, y_train, y_test)
