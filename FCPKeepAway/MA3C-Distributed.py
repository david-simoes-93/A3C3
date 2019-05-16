# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
#   tensorboard --logdir=worker_0:'./train_0'
#   tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3',worker_4:'./train_4',worker_5:'./train_5',worker_6:'./train_6',worker_7:'./train_7',worker_8:'./train_8',worker_9:'./train_9',worker_10:'./train_10',worker_11:'./train_11'


import argparse
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from FCPKeepAway.MA3CNetwork import AC_Network
from FCPKeepAway.MA3CSlave import Worker
from simulator_fcp.GymFCPKeepAway import GymFCPKeepAway, gym_fcp_kill_all
from simulator_fcp.Scenario import KeepAway

max_episode_length = 500
gamma = 0.9  # discount rate for advantage estimation and reward discounting
learning_rate = 1e-3
spread_messages = False
batch_size = 25

model_path = './dist_model'
display = False

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job"
)
parser.add_argument(
    "--slaves_per_url",
    type=str,
    default="1",
    help="Comma-separated list of maximum tasks within the job"
)
parser.add_argument(
    "--urls",
    type=str,
    default="localhost",
    help="Comma-separated list of hostnames"
)
parser.add_argument(
    "--comm_size",
    type=int,
    default=0,
    help="comm channels"
)
parser.add_argument(
    "--max_epis",
    type=int,
    default=200000,
    help="training steps"
)
parser.add_argument(
    "--demo",
    type=str,
    default="",
    help="demo folder"
)
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.task_index == 0:
    gym_fcp_kill_all()

number_of_agents = 3
comm_size = FLAGS.comm_size
amount_of_agents_to_send_message_to = number_of_agents - 1

if FLAGS.demo != "":
    model_path = FLAGS.demo
    FLAGS.task_index = 0
    FLAGS.slaves_per_url = "1"
    FLAGS.urls = "localhost"
    display = True
    learning_rate = 0
    FLAGS.max_epis += 1000
    batch_size = max_episode_length + 1

state_size = [15]
s_size_central = [8]
action_size = 5
env = GymFCPKeepAway(serverports=[3100+FLAGS.task_index*1000, 3200+FLAGS.task_index*1000])
critic_action = False
critic_comm = False

# Create a cluster from the parameter server and worker hosts.
hosts = []
for (url, max_per_url) in zip(FLAGS.urls.split(","), FLAGS.slaves_per_url.split(",")):
    for i in range(int(max_per_url)):
        hosts.append(url + ":" + str(2210 + i))
cluster = tf.train.ClusterSpec({"a3c": hosts})
server = tf.train.Server(cluster, job_name="a3c", task_index=FLAGS.task_index)

tf.reset_default_graph()

# Create a directory to save models and episode playback gifs
if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device(tf.train.replica_device_setter(worker_device="/job:a3c/task:%d" % FLAGS.task_index, cluster=cluster)):
    global_episodes = tf.contrib.framework.get_or_create_global_step()
    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    master_network = AC_Network(state_size, s_size_central, number_of_agents, action_size,
                                amount_of_agents_to_send_message_to * comm_size,
                                amount_of_agents_to_send_message_to * comm_size if spread_messages else comm_size,
                                'global', None)  # Generate global network

    # Master declares worker for all slaves
    for i in range(len(hosts)):
        print("Initializing variables for slave ", i)
        if i == FLAGS.task_index:
            worker = Worker(env, i, state_size, s_size_central,
                            action_size, number_of_agents, trainer, model_path,
                            global_episodes, amount_of_agents_to_send_message_to,
                            display=display and i == 0, comm=(comm_size != 0),
                            comm_size_per_agent=comm_size, spread_messages=spread_messages)
        else:
            Worker(None, i, state_size, s_size_central,
                   action_size, number_of_agents, trainer, model_path,
                   global_episodes, amount_of_agents_to_send_message_to,
                   display=display and i == 0, comm=(comm_size != 0),
                   comm_size_per_agent=comm_size, spread_messages=spread_messages)

print("Starting session", server.target, FLAGS.task_index)
hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_epis)]
with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),
                                       config=tf.ConfigProto(),
                                       save_summaries_steps=100,
                                       save_checkpoint_secs=600, checkpoint_dir=model_path, hooks=hooks) as mon_sess:
    print("Started session")
    worker.work(max_episode_length, gamma, mon_sess, batch_size=batch_size)
