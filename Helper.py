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


#test
#rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
#values = [-0.8408747, 0.316931, 0.48962486, 0.5403458, 0.25511515, 0.57507217, 0.6179354, 0.031220436, 0.05006236, 0.03478092, 0.035594642, 0.26584005, 0.55234975, 0.4835018, 0.5546427, 0.15869975, -0.41447937, 0.0050239563, 0.24489772, 0.24377179, 0.11405283, 0.19163871, -0.00011277199, -0.063069105, -0.066987514, 0.085493326]
#gamma = 0.9
#epsilon = 0.95
#advs1 = [1.0261126, 0.02373135, -0.103313655, -0.41074216, 0.1624498, -0.11893031, -0.68983704, -0.08616432, -0.11875953, -0.10274574, 0.10366139, 0.1312747, -0.21719813,
#         -0.08432338, -0.5118129, -0.63173115, 0.31900093, 0.11538399, -0.12550312, -0.24112424, -0.041578002, -0.2917402, -0.15664943, -0.09721966, 0.043931507]
#print(gae(gamma, epsilon, rewards, values))

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
