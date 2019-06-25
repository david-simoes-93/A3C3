# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
#   tensorboard --logdir=worker_0:'./train_0'
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7',worker_8:'./train_8',worker_9:'./train_9',worker_10:'./train_10',worker_11:'./train_11'


import argparse
import os
import threading
from time import sleep
import tensorflow as tf
from Navigation.MA3CNetwork import AC_Network
from Navigation.MA3CSlave import Worker
from simulator.GymNavigation import GymNav

max_episode_length = 500
gamma = 0.9  # discount rate for advantage estimation and reward discounting
learning_rate = 1e-4
spread_messages = False
batch_size = 25

load_model = False
model_path = './model'
display = False

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--num_slaves",
    type=int,
    default=1,
    help="Set number of available CPU threads"
)
parser.add_argument(
    "--num_agents",
    type=int,
    default=2,
    help="Set number of agents"
)
parser.add_argument(
    "--comm_size",
    type=int,
    default=0,
    help="comm channels"
)
parser.add_argument(
    "--critic",
    type=int,
    default=0,
    help="comm channels"
)
parser.add_argument(
    "--max_epis",
    type=int,
    default=150000,
    help="training steps"
)
parser.add_argument(
    "--demo",
    type=str,
    default="",
    help="demo folder"
)
parser.add_argument(
    "--comm_gaussian_noise",
    type=float,
    default=0,
    help="demo folder"
)
parser.add_argument(
    "--comm_delivery_failure_chance",
    type=float,
    default=0,
    help="demo folder"
)
parser.add_argument(
    "--comm_jumble_chance",
    type=float,
    default=0,
    help="demo folder"
)
parser.add_argument(
    "--param_search",
    type=str,
    default="40,relu,80,40",
    help="network configs. [size_of_comm_HL, activ_fn, size_of_actor_and_critic_HL1, size_of_actor_and_critic_HL2, ...]"
)
FLAGS, unparsed = parser.parse_known_args()
number_of_agents = FLAGS.num_agents
comm_size = FLAGS.comm_size
amount_of_agents_to_send_message_to = number_of_agents - 1

network_configs = FLAGS.param_search.split(",")
network_configs[0] = int(network_configs[0])
for i in range(2, len(network_configs)):
    network_configs[i] = int(network_configs[i])

if FLAGS.demo != "":
    load_model = True
    model_path = FLAGS.demo
    FLAGS.num_slaves = 1
    display = True
    learning_rate = 0
    FLAGS.max_epis += 1000
    batch_size = max_episode_length + 1

env = GymNav(number_of_agents=number_of_agents)
state_size = env.agent_observation_space
s_size_central = env.central_observation_space
action_size = env.agent_action_space
env.close()

critic_action = False
critic_comm = False
if FLAGS.critic == 1 or FLAGS.critic == 3:
    critic_action = True
if FLAGS.critic == 2 or FLAGS.critic == 3:
    critic_comm = True

tf.reset_default_graph()

# Create a directory to save models and episode playback gifs
if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    master_network = AC_Network(state_size, s_size_central, number_of_agents,
                                action_size, amount_of_agents_to_send_message_to * comm_size,
                                amount_of_agents_to_send_message_to * comm_size if spread_messages else comm_size,
                                'global',
                                None, critic_action=critic_action, critic_comm=critic_comm,
                                paramSearch=network_configs)  # Generate global network

    workers = []
    # Create worker classes
    for i in range(FLAGS.num_slaves):
        workers.append(Worker(GymNav(number_of_agents=number_of_agents), i, state_size, s_size_central,
                              action_size, number_of_agents, trainer, model_path,
                              global_episodes, amount_of_agents_to_send_message_to,
                              display=display and i == 0, comm=(comm_size != 0),
                              comm_size_per_agent=comm_size, spread_messages=spread_messages,
                              critic_action=critic_action, critic_comm=critic_comm,
                              comm_delivery_failure_chance=FLAGS.comm_delivery_failure_chance,
                              comm_gaussian_noise=FLAGS.comm_gaussian_noise,
                              comm_jumble_chance=FLAGS.comm_jumble_chance,
                              paramSearch=network_configs))

    saver = tf.train.Saver()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver, FLAGS.max_epis, batch_size)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)

        worker_threads.append(t)

    coord.join(worker_threads)
