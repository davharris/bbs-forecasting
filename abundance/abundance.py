import pandas as pd
import numpy as np
from scipy import stats, special

import tensorflow as tf
from tensorflow.python import math_ops
sess = tf.InteractiveSession()

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
%matplotlib inline

y_array = np.array(pd.read_csv("y.csv"))
n_Z = 5
decay_W = .1
decay_Z = .1



# Begin Tensorflow code

Y = tf.placeholder(tf.float32, shape=np.shape(y_array))
full_loss = tf.placeholder(tf.bool)

Z = tf.Variable(tf.truncated_normal([np.shape(y_array)[0],n_Z], stddev=1.0))
W   = tf.Variable(tf.truncated_normal([n_Z,np.shape(y_array)[1]], stddev=0.1 / np.sqrt(float(n_Z))))
W_p = tf.Variable(tf.truncated_normal([n_Z,np.shape(y_array)[1]], stddev=0.1 / np.sqrt(float(n_Z))))

b   = tf.cast(tf.Variable(tf.log(np.mean(y_array, axis=0))), tf.float32)
b_p = tf.cast(tf.Variable(tf.log(np.mean(y_array, axis=0))), tf.float32)

p_inflated = tf.nn.sigmoid(tf.matmul(Z,W_p) + b_p)

log_mu = tf.matmul(Z,W) + b

# log_mu, p_inflated, y
T = tf.pack([log_mu, p_inflated, Y], axis=2)


def log_sum_exp(x):
    a = tf.reduce_max(x)
    return a + tf.log(tf.reduce_sum((tf.exp(x - a))))

def zip_ll_row(T):
    log_mu_row     = T[:,0]
    p_inflated_row = T[:,1]
    y              = T[:,2]
    is_zero = tf.equal(y,0)
    is_nonzero = tf.not_equal(y,0)
    count_ll = -tf.nn.log_poisson_loss(log_mu_row, y, compute_full_loss = False)
    # When y!=0, we have 1-p_inflated _and_ the Poisson likelihood for y
    log_lik = tf.reduce_sum(
        tf.log(1-tf.boolean_mask(p_inflated_row, is_nonzero)) + 
            tf.boolean_mask(count_ll, is_nonzero)
    )
    # When y==0, we could have _either_ zero inflation (with probability p_inflated) _or_ the Poisson likelihood
    # for zero (with probability 1-p_inflated)
    log_lik += log_sum_exp(
        [tf.log(tf.boolean_mask(p_inflated_row, is_zero)), 
        tf.log(1-tf.boolean_mask(p_inflated_row, is_zero)) + 
            tf.boolean_mask(count_ll, is_zero)]
    )
    return log_lik

prediction_loss = -tf.reduce_mean(tf.map_fn(zip_ll_row, T))
regularization_loss = tf.reduce_sum(tf.pow(W * decay_W, 2)) + tf.reduce_sum(tf.pow(Z * decay_Z, 2))
loss = prediction_loss + regularization_loss

train_step = tf.train.AdamOptimizer().minimize(loss)



# End model description; begin fitting

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(10):
  sess.run(train_step, feed_dict={Y:y_array})
  
sess.run(prediction_loss, feed_dict={Y:y_array})