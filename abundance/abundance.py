import pandas as pd
import numpy as np
from scipy import stats, special

import tensorflow as tf
from tensorflow.python import math_ops
sess = tf.InteractiveSession()

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

# Helper functions & negative binomial loss
def logit(x):
    return np.log(x) - np.log(1-x)
    

def log1p(x):
    """log(1+x) when x is close to zero; based on npymath.npy_log1p"""
    u = 1. + x
    d = u - 1.
    value = tf.log(u) * x / d
    # return x if x is +Inf
    value = tf.select(tf.is_inf(x), x, value)
    # return 0 if x is 1
    value = tf.select(tf.equal(x, 1.), tf.zeros_like(value), value)
    return value


def xlog1py(x,y):
    """return x*log1p(y), unless x==0; based on scipy.special.xlog1py"""
    value = x * log1p(y)
    # Return zero when x==0
    return tf.select(tf.equal(x, 0.), tf.zeros_like(value), value)
    

def nbinom_ll(x, n, p):
    """modified from scipy.stats.nbinom._lpmf"""
    coeff = tf.lgamma(n+x) - tf.lgamma(x+1) - tf.lgamma(n)
    return coeff + n*tf.log(p) + xlog1py(x, -p)


def nbinom2_ll(x, mu, size):
    p = size / (mu + size)
    return nbinom_ll(x, size, p)
    

# Data & Settings
y_array = np.array(pd.read_csv("y.csv"))
n_Z = 10
decay_W = .01
decay_Z = .1


# Main TF model code
Y = tf.placeholder(tf.float32, shape=np.shape(y_array))
full_loss = tf.placeholder(tf.bool)

Z = tf.Variable(tf.truncated_normal([np.shape(y_array)[0],n_Z], stddev=1.0))
W   = tf.Variable(tf.truncated_normal([n_Z,np.shape(y_array)[1]], stddev=0.1 / np.sqrt(float(n_Z))))
W_inflated = tf.Variable(tf.truncated_normal([n_Z,np.shape(y_array)[1]], stddev=0.1 / np.sqrt(float(n_Z))))
W_size = tf.Variable(tf.truncated_normal([n_Z,np.shape(y_array)[1]], stddev=0.1 / np.sqrt(float(n_Z))))

b   = tf.cast(tf.Variable(tf.log(np.mean(y_array, axis=0))), tf.float32)
b_inflated = tf.cast(tf.Variable(logit(np.mean(y_array == 0, axis=0))), tf.float32)
b_size = tf.ones_like(b_inflated)

p_inflated = tf.nn.sigmoid(tf.matmul(Z,W_inflated) + b_inflated)
size = tf.exp(tf.matmul(Z,W_size) + b_size)
log_mu = tf.matmul(Z,W) + b

def log_sum_exp(x, y):
    a = tf.maximum(x, y)
    return a + tf.log(tf.exp(x - a) + tf.exp(y - a))

def zip_ll(log_mu, size, p_inflated, y):
    is_zero = tf.equal(y,0)
    is_nonzero = tf.not_equal(y,0)
    count_ll = nbinom2_ll(y, tf.exp(log_mu), size)    
    # When y!=0, we have 1-p_inflated _and_ the Poisson likelihood for y
    log_lik = tf.reduce_sum(
        tf.log(1-tf.boolean_mask(p_inflated, is_nonzero)) + 
            tf.boolean_mask(count_ll, is_nonzero)
    )
    # When y==0, we could have _either_ zero inflation (with probability p_inflated) _or_ the Poisson likelihood
    # for zero (with probability 1-p_inflated)
    log_lik += tf.reduce_sum(
        log_sum_exp(
        tf.log(tf.boolean_mask(p_inflated, is_zero)), 
        tf.log(1-tf.boolean_mask(p_inflated, is_zero)) + 
            tf.boolean_mask(count_ll, is_zero)
        )
    )
    return log_lik / tf.cast(tf.shape(y)[0], tf.float32)


prediction_loss = -zip_ll(log_mu, size, p_inflated, y=Y)

regularization_loss = tf.reduce_sum(tf.pow(Z * decay_Z, 2)) +tf.reduce_sum(tf.pow(W * decay_W, 2)) + tf.reduce_sum(tf.pow(W_size * decay_W, 2)) + tf.reduce_sum(tf.pow(W_inflated * decay_W, 2))
    
loss = prediction_loss + regularization_loss

train_step = tf.train.AdamOptimizer().minimize(loss)


# Session incantations 
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# Fit the model
print(sess.run(prediction_loss, feed_dict={Y:y_array}))
print(sess.run(regularization_loss, feed_dict={Y:y_array}))
n_steps = 0
for i in range(1000):
    sess.run(train_step, feed_dict={Y:y_array})
    n_steps += 1
print(sess.run(prediction_loss, feed_dict={Y:y_array}))


np.savetxt("Z.csv", sess.run(Z, feed_dict={Y:y_array}), delimiter=",")
np.savetxt("log_y_hat.csv", sess.run(log_mu, feed_dict={Y:y_array}), delimiter=",")
np.savetxt("p_zero.csv", sess.run(p_inflated, feed_dict={Y:y_array}), delimiter=",")