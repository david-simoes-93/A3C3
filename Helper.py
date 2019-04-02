import numpy as np
import scipy.signal
import tensorflow as tf
import math
import random


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        # print(from_var, to_var)
        op_holder.append(to_var.assign(from_var))
    return op_holder


def one_hot_encoding(arr, max):
    one_hot = [0]*(len(arr)*max)
    for index in range(len(arr)):
        one_hot[index*max+arr[index]]=1
    return one_hot

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# From GAE, defined in pg4, between eqs.9 and 10
def get_sigma(discount_factor, rewards, value):
    sigmas = []
    for t in range(len(rewards)):
        sigmas.append(rewards[t] + discount_factor*value[t+1]-value[t])
    return sigmas


# GAE
def gae(discount_factor, epsilon, rewards, values):
    advantages = []
    dfe = discount_factor*epsilon
    sigmas = get_sigma(discount_factor, rewards, values)
    for t in range(len(rewards)):
        curr_gae = 0
        for l in range(len(rewards)-t):
            curr_gae += (dfe**l)*sigmas[t+l][0]
        advantages.append(curr_gae)

    return advantages


# GAE with Epsilon=0
def gae_0(discount_factor, rewards, values):
    advantages = rewards + discount_factor * values[1:] - values[:-1]
    return advantages


# GAE with Epsilon=1
def gae_1(discount_factor, rewards, values):
    advantages = []
    for t in range(len(rewards)):
        curr_gae = 0
        for l in range(len(rewards)-t):
            curr_gae += (discount_factor**l)*rewards[t+l]
        advantages.append(curr_gae-values[t][0])
    return advantages


# Regular advantages
def adv(discounted_rewards, values):
    advantages = discounted_rewards - values[:-1]
    return advantages


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer



# just some empty arrays
def get_empty_loss_arrays(size):
    v_l = np.empty(size)
    p_l = np.empty(size)
    e_l = np.empty(size)
    g_n = np.empty(size)
    v_n = np.empty(size)
    return v_l, p_l, e_l, g_n, v_n
