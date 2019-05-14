import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from custom_losses import custom_focal_loss
from preprocessing import prepare_real_test_data


# losses.custom_focal_loss=custom_focal_loss

def predict(model_path, nn_type, X_test, output):
    loaded_model = load_model(model_path, custom_objects={'custom_focal_loss': custom_focal_loss})
    predictions = None
    print("Loaded model from disk")
    print(loaded_model.summary())
    if nn_type == 'deep_supervised_autoencoder':
        predictions = np.rint(np.asarray(loaded_model.predict(X_test)[1]))
    else:
        predictions = np.asarray(loaded_model.predict_classes(X_test))
    pd.DataFrame(predictions, columns=['fraud'],dtype='int32').to_csv(output, index=False)


X_test = prepare_real_test_data()
# predict('../models/deep-supervised-autoencoder/deep_supervised_autoencoder_6_4_lr_001_tanh_0.h5',
#         'deep_supervised_autoencoder', X_test, '../submissions/01_TU_Darmstadt_1.csv')
# predict('../models/deep-supervised-autoencoder/deep_supervised_autoencoder_6_4_lr_001_tanh_1.h5',
#         'deep_supervised_autoencoder', X_test, '../submissions/02_TU_Darmstadt_1.csv')

for i in range(8):
    predict('../models/deep-supervised-autoencoder/deep_supervised_autoencoder_with_fold_6_4_lr_001_custom_focal_loss_tanh_'+str(i)+'.h5',
        'deep_supervised_autoencoder', X_test, '../submissions/'+str(i)+'_TU_Darmstadt_1.csv')



