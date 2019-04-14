import numpy as np
import tensorflow as tf

# 自适应颜色反卷积算法，优化染色外观矩阵SCA

# rgb mode
init_varphi = np.asarray([[0.6060, 1.2680, 0.7989],
                          [1.2383, 1.2540, 0.3927]])

def acd_model(input_od, lambda_p=0.002, lambda_b=10, lambda_e=1, eta=0.6, gamma=0.5):
    """
    Stain matrix estimation via method of 
    "Yushan Zheng, et al., Adaptive Color Deconvolution for Histological WSI Normalization."

    """
    # alpha beta为六个自由度变量，因w是一个对角矩阵
    alpha = tf.Variable(init_varphi[0], dtype='float32')
    beta = tf.Variable(init_varphi[1], dtype='float32')
    w = [tf.Variable(1.0, dtype='float32'), tf.Variable(1.0, dtype='float32'), tf.constant(1.0)]

    # SCA 染色外观矩阵，即M，CD 颜色反卷积矩阵，即D 二者互为逆
    sca_mat = tf.stack((tf.cos(alpha) * tf.sin(beta), tf.cos(alpha) * tf.cos(beta), tf.sin(alpha)), axis=1)
    cd_mat = tf.matrix_inverse(sca_mat)

    # 染色量 s = od*cd*w
    s = tf.matmul(input_od, cd_mat) * w
    h, e, b = tf.split(s, (1, 1, 1), axis=1)

    # 目标函数、染色分量比函数、染色总量控制函数
    l_p1 = tf.reduce_mean(tf.square(b))
    l_p2 = tf.reduce_mean(2 * h * e / (tf.square(h) + tf.square(e)))
    l_b = tf.square((1 - eta) * tf.reduce_mean(h) - eta * tf.reduce_mean(e))
    l_e = tf.square(gamma - tf.reduce_mean(s))

    # 最终目标函数，梯度下降优化
    objective = l_p1 + lambda_p * l_p2 + lambda_b * l_b + lambda_e * l_e
    target = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(objective)

    return target, cd_mat, w