import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def custom_focal_loss(y_true, y_pred):
    gamma = 2
    alpha = 1 / 5
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))



def custom_binary_crossentropy():
    weights = np.ones((2, 2))
    weights[0, 0] = 1e-7
    weights[0, 1] = 5 / 6
    weights[1, 0] = 1 / 6
    weights[1, 1] = 1e-7

    def loss_function(y_true, y_pred):
        # y_pred_round=K.round(y_pred)
        loss = -(y_true * (K.log(y_pred) * weights[1, 0] + K.log(1 - y_pred) * weights[1, 1])
                 + (1 - y_true) * (K.log(1 - y_pred) * weights[0, 1] + K.log(y_pred) * weights[0, 0]))

        return K.mean(loss, axis=-1)
        # return K.binary_crossentropy(y_true,y_pred)

    return loss_function

# def custom_categorical_crossentropy(y_true, y_pred):
#     from itertools import product
#     weights = np.ones((2, 2))
#     weights[0, 0] = 1e-7
#     weights[0, 1] = 5/6
#     weights[1, 0] = 1/6
#     weights[1, 1] = 1e-7
#     nb_cl = len(weights)
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=1)
#     y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
#     y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#         final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#     cross_ent = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
#     return cross_ent * final_mask


def focal_loss(gamma=2., alpha=0.5):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed
