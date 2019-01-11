import numpy as np
import random
from time import sleep

from simulator.GymTraffic import GymTraffic


def output_mess_to_input_mess(message, comm_map):
    curr_comm = []
    no_mess = [0]
    # for agent_state in states:
    for j, agent_state in enumerate(comm_map):
        curr_agent_comm = []
        # print(agent_state)
        for neighbor in agent_state:
            if neighbor != -1:
                # print("message from ", neighbor, "to", j)
                curr_agent_comm.extend(message[neighbor])
                # curr_agent_comm[-1] = 0
            else:
                curr_agent_comm.extend(no_mess)
                # curr_agent_comm[-1] = 1
        curr_comm.append(curr_agent_comm)

    return curr_comm


qvals = {'[0, -1, 0]': [0.45207421405144876, 0.2195192970555739],       # Outside map, whatever
         '[-1, -1, 0]': [-1.689191969393746, 0.27975360565097374],     # Alone, go
         '[1, -1, 0]': [-1.5650678504017481, 0.32636727016613376],     # Alone, go

         '[1, 1, -1]': [-2.744392899503402, -1.063742767464318],       # Prio, crash, go
         '[1, 1, 1]': [-0.946269003494488, -0.3367585135819862],      # Prio, no crash, go

         '[-1, 1, 1]': [-1.1388908032335308, -6.279607263974365],     # No Prio, crash, wait
         '[-1, 1, -1]': [-1.8910108101143843, -0.45601986940866585]}   # No Prio, no crash, go

a = {'[-1, -1, 0]': [-0.5533549480267829, 0.4682110650930802],
     '[1, -1, 0]': [-0.1963372844445958, 0.5824676993884524],

     '[-1, 1, -1]': [-1.1774350886902198, 0.3861796002335611],
     '[-1, 1, 0]': [-0.4402617388016025, -1.7387111221843217],
     '[-1, 1, 1]': [-0.5325489954520771, -5.491238674946438],

     '[1, 1, -1]': [-1.6303601192009438, 0.3736056557074792],
     '[1, 1, 0]': [-1.2060258053734292, 0.20186374488527753],
     '[1, 1, 1]': [-0.6827812701653839, 0.09448693262311685]}


qvals = {}

number_of_agents = 40
max_actions = 2
env = GymTraffic(number_of_agents=number_of_agents)
display = False
amount_of_agents_to_send_message_to = 3
message_size = 1
max_episode_length = 500
comm_delivery_failure_chance = 0.5
exploration_rate = 1
learning_rate = 0.05

while True:
    for episode_count in range(30000):
        if episode_count % 100 == 0:
            print(episode_count, "lambda:", exploration_rate)
        exploration_rate -= (2 / 30000)
        if exploration_rate < 0.05:
            exploration_rate = 0.05

        episode_comm_maps = [[] for _ in range(number_of_agents)]
        episode_reward = 0
        episode_step_count = 0

        # start new epi
        current_screen, info = env.reset()
        for i in range(number_of_agents):
            episode_comm_maps[i].append(current_screen[i][1:4])

            # replace state to just show whether cars were there or not, and not which cars
            for neighbor_index in range(1, 4):
                if current_screen[i][neighbor_index] >= 0:
                    current_screen[i][neighbor_index] = 1

        if display:
            env.render()

        curr_comm = [[] for _ in range(number_of_agents)]
        for curr_agent in range(number_of_agents):
            for from_agent in range(amount_of_agents_to_send_message_to):
                curr_comm[curr_agent].extend([0] * message_size)
        state_key = [str([current_screen1[0]] + [current_screen1[3]] + [curr_comm1[2]]) for current_screen1, curr_comm1
                     in zip(current_screen, curr_comm)]
        for index in range(number_of_agents):
            if state_key[index] not in qvals.keys():
                qvals[state_key[index]] = [0, 0]

        for episode_step_count in range(max_episode_length):
            actions = []
            message = []
            for index in range(number_of_agents):
                message.append([1] if current_screen[index][0] == 1 else [-1])
                if random.random() < exploration_rate:
                    actions.append(0 if random.random() < .5 else 1)
                else:
                    actions.append(0 if qvals[state_key[index]][0] > qvals[state_key[index]][1] else 1)

            previous_state_key = state_key

            # Watch environment
            current_screen, reward, terminal, _ = env.step(actions)
            episode_reward += sum(reward)

            this_turns_comm_map = []
            for i in range(number_of_agents):
                # 50% chance of no comms
                surviving_comms = current_screen[i][1:4]
                for index in range(len(surviving_comms)):
                    if random.random() < comm_delivery_failure_chance:  # chance of failure comms
                        surviving_comms[index] = -1
                episode_comm_maps[i].append(surviving_comms)
                this_turns_comm_map.append(surviving_comms)
                # replace state to just show whether cars were there or not, and not which cars
                for neighbor_index in range(1, 4):
                    if current_screen[i][neighbor_index] >= 0:
                        current_screen[i][neighbor_index] = 1
            curr_comm = output_mess_to_input_mess(message, this_turns_comm_map)

            state_key = [str([current_screen1[0]] + [current_screen1[3]] + [curr_comm1[2]]) for
                         current_screen1, curr_comm1 in
                         zip(current_screen, curr_comm)]

            for index in range(number_of_agents):
                if state_key[index] not in qvals.keys():
                    qvals[state_key[index]] = [0, 0]

            if display:
                env.render()
                sleep(0.2)

            if terminal:
                for index in range(number_of_agents):
                    qvals[previous_state_key[index]][actions[index]] = \
                        (1 - learning_rate) * qvals[previous_state_key[index]][actions[index]] + \
                        learning_rate * (reward[index])
                break
            for index in range(number_of_agents):
                qvals[previous_state_key[index]][actions[index]] = \
                    (1 - learning_rate) * qvals[previous_state_key[index]][actions[index]] + \
                    learning_rate * (reward[index] + 0.9 * max(qvals[state_key[index]]))

        if episode_count % 100 == 0:
            print("0ver ", episode_step_count, episode_reward)
        if episode_count % 1000 == 0:
            print(qvals)

    print(qvals)
    # display = True
