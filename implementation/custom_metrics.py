from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
import  numpy as np

class CustomMetrics(Callback):
    def __init__(self, nn_type=None):
        self.nn_type=nn_type

    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.scores=[]
        self.avg_scores=[]

    # def on_epoch_end(self, epoch, logs={}):
    #     #score = np.asarray(self.model.predict(self.validation_data[0]))
    #     threshold=0.5
    #     predict = np.round(np.asarray(self.model.predict(self.validation_data[0])>threshold))
    #     label_predict=[]
    #     for i in range(len(predict)):
    #         if predict[i][0]==1:
    #             label_predict.append(0)
    #         else:
    #             label_predict.append(1)
    #     label_predict=np.array(label_predict)
    #     #predict = np.asarray(self.model.predict_classes(self.validation_data[0]))
    #     targ = self.validation_data[1]
    #     label_targ=[]
    #     for i in range(len(targ)):
    #        if targ[i][0]==1:
    #            label_targ.append(0)
    #        else:
    #            label_targ.append(1)
    #     label_targ=np.array(label_targ)
    #
    #     cf_matrix=confusion_matrix(label_targ, label_predict)
    #     s=cf_matrix[1][1]*5-cf_matrix[0][1]*25-cf_matrix[1][0]*5
    #     avg_score=s/(cf_matrix[1][1]+cf_matrix[0][1]+cf_matrix[1][0])
    #     self.confusion.append(confusion_matrix(label_targ, label_predict).ravel())
    #     self.precision.append(precision_score(label_targ, label_predict))
    #     self.recall.append(recall_score(label_targ, label_predict))
    #     self.f1s.append(f1_score(label_targ, label_predict))
    #     self.kappa.append(cohen_kappa_score(label_targ, label_predict))
    #     self.scores.append(s)
    #     self.avg_scores.append(avg_score)
    def on_epoch_end(self, epoch, logs={}):
        #score = np.asarray(self.model.predict(self.validation_data[0]))
        threshold=0.5
        predict=None
        targ=None
        if self.nn_type=='deep_supervised_autoencoder':
            predict = np.rint(np.asarray(self.model.predict(self.validation_data[1])[1]))
            targ = self.validation_data[2]
        else:
            predict = np.rint(np.asarray(self.model.predict_classes(self.validation_data[0])))
            targ = self.validation_data[1]
        #print(predict)
        cf_matrix=confusion_matrix(targ, predict)
        s=cf_matrix[1][1]*5-cf_matrix[0][1]*25-cf_matrix[1][0]*5
        avg_score=s/(cf_matrix[1][1]+cf_matrix[0][1]+cf_matrix[1][0])
        self.confusion.append(confusion_matrix(targ, predict).ravel())
        self.precision.append(precision_score(targ, predict))
        self.recall.append(recall_score(targ, predict))
        self.f1s.append(f1_score(targ, predict))
        self.kappa.append(cohen_kappa_score(targ, predict))
        self.scores.append(s)
        self.avg_scores.append(avg_score)
