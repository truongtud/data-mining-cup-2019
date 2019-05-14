import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix

x = tf.constant([0.1,0.9, 2.5, 2.3, 1.5, -4.5])
y=tf.constant([[1,0,0],[1,0,1]])


print((y[:,1]))
weights = np.ones((2, 2))
np.array([[1,0,0],[1,0,1]])
weights[0, 1] = 25
weights[1, 0] = 5
#final_mask = K.ones_like(y_pred[:, 0])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    v = sess.run(y[:,0])
    print(v)  # will show you your variable.
cm=confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0])
print(cm)
tn, fp, fn, tp = cm.ravel()
