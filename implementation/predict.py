from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import model_from_json,load_model
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from preprocessing import *
import  numpy as np

loaded_model=load_model('../models/pre-trained/model_autoencoder_relu_6_4_lr_00001_tanh.h5')
print("Loaded model from disk")
print(loaded_model.summary())

labeled_df = read_data('../DMC_2019_task/train.csv')
unlabled_df=read_data('../DMC_2019_task/test.csv')
labeled_X= labeled_df.iloc[:, 0:9]
labeled_y=labeled_df.iloc[:,9]

labeled_X=one_hot_trust_level(labeled_X)
unlabled_X =one_hot_trust_level(unlabled_df)
X=pd.concat([labeled_X,unlabled_X])
scaler=StandardScaler()
scaler.fit(X)
labeled_X= scale(labeled_X, scaler)
unlabled_X=scale(unlabled_X,scaler)

decoded_X=loaded_model.predict(labeled_X)
mse=np.mean(np.square(abs(labeled_X-decoded_X)),axis=1)

for i in range(len(labeled_X)):
    print('%s: %f'%(mse[i],labeled_y[i]))

mse_1 = np.mean(np.power(labeled_X - decoded_X, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse_1,
                         'True_class': labeled_y})
print(error_df.describe())
print(error_df.Reconstruction_error.values)

false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
print(thresholds)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=1, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=1)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
