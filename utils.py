import tensorflow as tf
import os
import random
import numpy as np

def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))
    
def ccgan_D_real_loss(y, y_hat): #y [n], y_hat [1 6 8 n]
    
    inner_product = tf.tensordot(y, y_hat, [[0],[3]]) # [1 6 8]
    inner_product = tf.expand_dims(inner_product, axis = -1) #[1 6 8 1]
    oneslike = tf.ones_like(inner_product) #[1 6 8 1]
    
    return l2_loss(y = oneslike, y_hat = inner_product)

def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))


