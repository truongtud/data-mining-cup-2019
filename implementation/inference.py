from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tensorflow.keras.models import model_from_json
from preprocessing import *
import  numpy as np
json_file = open('model_deep_nn.json', 'r')
loaded_model_json = json_file.read()
#json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_deep_nn.h5")
print("Loaded model from disk")

unlabled_X = scale(one_hot_trust_level(read_data('../DMC_2019_task/test.csv')),RobustScaler())

predicted=loaded_model.predict(unlabled_X)

for i in range(len(unlabled_X)):
    print('%s: %f'%(i+1,predicted[i]))