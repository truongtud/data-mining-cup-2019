from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
import  numpy as np



def evaluate(model,X_test,y_test):
    predict = np.asarray(model.predict_classes(X_test))
    targ = y_test
    cf_matrix=confusion_matrix(targ, predict)
    s=cf_matrix[1][1]*5-cf_matrix[0][1]*25-cf_matrix[1][0]*5
    avg_score=s/(cf_matrix[1][1]+cf_matrix[0][1]+cf_matrix[1][0])
    print(str(cf_matrix) + '---->' + str(avg_score))