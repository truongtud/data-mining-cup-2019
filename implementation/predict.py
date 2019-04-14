from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import model_from_json

from preprocessing import *
import  numpy as np
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

df = read_data('../DMC_2019_task/train.csv')
fraud_df = fraud_instances(df)
groupByTrustLevel = fraud_instances_groupby(fraud_df, 'trustLevel')

X, y = training_data(df)
X = scale(X, StandardScaler())

decoded_X=loaded_model.predict(X)
mse=np.sum(np.square(abs(X-decoded_X)),axis=1)
for i in range(len(X)):
    print('%s: %f'%(mse[i],y[i]))
