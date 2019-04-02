import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from Helper import normalized_columns_initializer


# Actor-Critic Network
class AC_NetworkMPE:
    def __init__(self, s_size, s_size_central, number_of_agents, a_size, comm_size_input, comm_size_output, scope, trainer,
                 critic_comm=False, reduce_type="max"):
        with tf.variable_scope(scope):
            print("Scope", scope)
            if critic_comm:
                central_input_size = [s_size_central[0], s_size_central[-1] + comm_size_input]
            else:
                central_input_size = s_size_central

            self.inputs = tf.placeholder(shape=[None, ] + s_size, dtype=tf.float32)
            self.inputs_central = tf.placeholder(shape=[None, ] + central_input_size, dtype=tf.float32)
            central_input_reshaped = tf.reshape(self.inputs_central, [-1, 1, central_input_size[-1], 1])
            self.inputs_comm = tf.placeholder(shape=[None, comm_size_input], dtype=tf.float32)

            flattened_inputs = tf.contrib.layers.flatten(self.inputs)
            self.flattened_inputs_with_comm = tf.concat([flattened_inputs, self.inputs_comm], 1)

            hidden_comm = slim.fully_connected(flattened_inputs, 40,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.nn.relu)

            hidden = slim.fully_connected(self.flattened_inputs_with_comm, 80,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.nn.relu)

            hidden2 = slim.fully_connected(hidden, 40,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.nn.relu)

            # Convolutional Layer #1
            filters1 = 80
            conv1 = tf.layers.conv2d(central_input_reshaped, filters=filters1, kernel_size=[1, central_input_size[-1]],
                                     padding="valid", kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            conv1_reshaped = tf.reshape(conv1, [-1, number_of_agents, filters1, 1])

            # Convolutional Layer #2
            filters2 = 40
            conv2 = tf.layers.conv2d(inputs=conv1_reshaped, filters=filters2, kernel_size=[1, filters1],
                                     padding="valid", kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
            conv2_reshaped = tf.reshape(conv2, [-1, number_of_agents, filters2])

            # Reduction layer
            if reduce_type == "max":
                reduced_layer = tf.reduce_max(conv2_reshaped, axis=1)
            elif reduce_type == "mean":
                reduced_layer = tf.reduce_mean(conv2_reshaped, axis=1)
            elif reduce_type == "softmax":
                self.vari = tf.Variable(normalized_columns_initializer(1.0)([1, filters2]))
                softmax = tf.nn.softmax(tf.multiply(conv2_reshaped, self.vari), axis=1)
                reduced_layer = tf.reduce_sum(tf.multiply(softmax, conv2_reshaped), axis=1)
            else:
                print("No reduce type specified:", reduce_type)
                reduced_layer = tf.reduce_max(conv2_reshaped, axis=1)

            self.value = slim.fully_connected(reduced_layer, 1, activation_fn=None,
                                                 weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=normalized_columns_initializer(1.0))
            self.policy = slim.fully_connected(hidden2, a_size, activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=normalized_columns_initializer(0.01))
            if comm_size_output != 0:
                self.message = slim.fully_connected(hidden_comm, comm_size_output,
                                                    activation_fn=tf.nn.tanh,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.contrib.layers.xavier_initializer())
            else:
                self.message = slim.fully_connected(hidden_comm, comm_size_output)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)                 # Index of actions taken
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)    # 1-hot tensor of actions taken
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)              # Target Value
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)            # temporary difference (R-V)

                self.log_policy = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))         # avoid NaN with clipping when value in policy becomes zero
                self.responsible_outputs = tf.reduce_sum(self.log_policy * self.actions_onehot, [1]) # Get policy*actions influence
                self.r_minus_v = self.target_v - tf.reshape(self.value, [-1])               # difference between target value and actual value

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.r_minus_v))            # same as tf.nn.l2_loss(r_minus_v)
                self.entropy = - tf.reduce_sum(self.policy * self.log_policy)               # policy entropy
                self.policy_loss = -tf.reduce_sum(self.responsible_outputs * self.advantages)   # policy loss

                # loss of message
                self.target_message = tf.placeholder('float32', [None, comm_size_output], name='target_message')
                self.loss_m = tf.reduce_mean(tf.square(self.target_message - self.message), name='loss_m')

                # Learning rate for Critic is half of Actor's, so value_loss/2 + policy loss
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, self.local_vars)
                self.var_norms = tf.global_norm(self.local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # gradients of "loss" wrt message
                self.gradients_q_message = tf.gradients(self.policy_loss, self.inputs_comm)
                # gradients of "target message" wrt weights
                self.gradients_m_weights = tf.gradients(self.loss_m, self.local_vars)
                grads_m, self.grad_norms_m = tf.clip_by_global_norm(self.gradients_m_weights, 40.0)

                # Apply local gradients to global network
                self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, self.global_vars))
                self.apply_grads_m = trainer.apply_gradients(zip(grads_m, self.global_vars))
