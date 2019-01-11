import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os


def merge(ls_x, ls_y):
    l2_x, l2_y = [], []

    # ensure longest list on X is on index 0
    last_elements = [l[-1] for l in ls_x]
    index_of_longest_list = last_elements.index(max(last_elements))
    temp_list = ls_x[0]
    ls_x[0] = ls_x[index_of_longest_list]
    ls_x[index_of_longest_list] = temp_list
    temp_list = ls_y[0]
    ls_y[0] = ls_y[index_of_longest_list]
    ls_y[index_of_longest_list] = temp_list

    while max([len(l) for l in ls_x]) > 0:
        # find list with lowest X
        # print(ls_x[0][0])
        min_index, min_val = 0, ls_x[0][0]
        for i in range(1, len(ls_x)):
            if len(ls_x[i]) != 0:
                if ls_x[i][0] <= min_val:
                    min_index = i
                    min_val = ls_x[i][0]

        l2_x.append(ls_x[min_index][0])
        l2_y.append(ls_y[min_index][0])
        ls_x[min_index] = ls_x[min_index][1:]
        ls_y[min_index] = ls_y[min_index][1:]

    return l2_x, l2_y


# returns mean and std of a single list over groups of n elements
def mean_std(l, n, std_multiplier=1):
    l2_mean, l2_std = [], []
    i = 0
    while i + n < len(l):
        buffer = []
        for j in range(n):
            buffer.append(l[i])
            i += 1
        l2_mean.append(np.mean(buffer))
        l2_std.append(np.std(buffer) * std_multiplier)
    return l2_mean, l2_std


# averages the value of a single list over groups of n elements
def mean(l, n):
    l2 = []
    i = 0
    while i + n < len(l):
        buffer = 0
        for j in range(n):
            buffer += l[i]
            i += 1
        l2.append(buffer / n)
    return l2


trains = 3
avg_value = 100
base_dir = "/home/david/GitHub/DeepComm/"
f = plt.figure()
color = ['red', 'blue', 'purple', 'yellow', 'green', 'pink']
color_comp = ['red', 'blue', 'purple', 'green']

###############################################

# Blind Group Up Rod
"""
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials_for_acs = {0: {0: 0, 1: 0},
                           5: {0: 0, 1: 1, 2: 2, 3: 0},
                           10: {0: 2, 1: 1, 2: 2, 3: 2},
                           20: {0: 1, 1: 0, 2: 0, 3: 2}}
for server in ["TB-Retina-dcomm6"]:
    for comm in [0, 5, 10, 20]:
        ac = 0
        for trial in range(5):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir + '/Trials/TB-Retina-dcomm7.bgu/BlindGroupUp-' + str(comm) + 'Comm-Trial' + str(trial) + \
                      '/train_' + str(train)
                if not os.path.isdir(dir):
                    print(dir + " not found!")
                    continue

                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            if len(xs) == 0:
                print("No trial found")
                continue
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial], label="Trial " + str(trial))

            if trial == approved_trials_for_acs[comm][ac]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

        # plt.show()
        plt.xlim(0, 200000)  # set the xlim to xmin, xmax
        plt.ylim(10, 100)
        plt.legend()
        plt.savefig(base_dir + '/BlindGroupUp-A3C3-' + str(comm) + 'Comm-AC' + str(ac) + '.pdf',
                    bbox_inches='tight')
        plt.clf()

        for ac in range(1, 4):
            for trial in range(3):
                xs, ys = [], []
                for train in range(trains):
                    dir = base_dir + server + '/BlindGroupUp-' + str(comm) + 'Comm-Trial' + str(trial) + \
                          '-AC' + str(ac) + '/train_' + str(train)
                    if not os.path.isdir(dir):
                        print(dir + " not found!")
                        continue

                    print("Reading", dir)
                    event_acc = event_accumulator.EventAccumulator(dir)
                    event_acc.Reload()

                    w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                    xs.append(step_nums)
                    ys.append(vals)
                if len(xs) == 0:
                    print("No trial found")
                    continue
                x, y = merge(xs, ys)
                mean_x = mean(x, avg_value)
                mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
                upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
                plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
                plt.plot(mean_x, mean_y, color=color[trial], label="Trial " + str(trial))

                if trial == approved_trials_for_acs[comm][ac]:
                    approved_mean.append(mean_y)
                    approved_up.append(upper_std)
                    approved_down.append(lower_std)
                    approved_x.append(mean_x)

            # plt.show()
            plt.xlim(0, 200000)  # set the xlim to xmin, xmax
            plt.ylim(10, 100)
            plt.legend()
            plt.savefig(base_dir + '/BlindGroupUp-A3C3-' + str(comm) + 'Comm-AC' + str(ac) + '.pdf',
                        bbox_inches='tight')
            plt.clf()

        for i, x, y, u, d in zip(range(len(approved_x)), approved_x, approved_mean, approved_up, approved_down):
            plt.fill_between(x, u, d, color=color[i], alpha=0.2)
            plt.plot(x, y, color=color[i], linewidth=0.5, label="AC " + str(i))
        plt.xlim(0, 200000)  # set the xlim to xmin, xmax
        plt.ylim(10, 100)
        plt.legend()
        plt.savefig(base_dir + '/BlindGroupUp-A3C3-Comm' + str(comm) + '.pdf', bbox_inches='tight')
        plt.clf()
        approved_mean, approved_up, approved_down, approved_x = [], [], [], []
exit()"""


###############################################

"""# Traffic Rod
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials_for_acs = {1: {0: 0, 1: 0, 2: 0, 3: 0}}
for server in ["Lily-Traffic"]:
    for comm in [1]:
        for ac in range(4):
            for trial in range(1):
                xs, ys = [], []
                for train in range(trains):
                    dir = base_dir + server + '/TrafficAC' + str(ac) + '/train_' + str(train)
                    if not os.path.isdir(dir):
                        print(dir + " not found!")
                        continue

                    print("Reading", dir)
                    event_acc = event_accumulator.EventAccumulator(dir)
                    event_acc.Reload()

                    w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                    xs.append(step_nums)
                    ys.append(vals)

                if len(xs) == 0:
                    print("No trial found")
                    continue
                x, y = merge(xs, ys)
                mean_x = mean(x, avg_value)
                mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
                upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
                plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
                plt.plot(mean_x, mean_y, color=color[trial], label="Trial " + str(trial))

                if trial == approved_trials_for_acs[comm][ac]:
                    approved_mean.append(mean_y)
                    approved_up.append(upper_std)
                    approved_down.append(lower_std)
                    approved_x.append(mean_x)

            # plt.show()
            plt.xlim(0, 5000)  # set the xlim to xmin, xmax
            plt.ylim(0, 60)
            plt.legend()
            plt.savefig(base_dir + '/Traffic-A3C3-' + str(comm) + 'Comm-AC' + str(ac) + '.pdf',
                        bbox_inches='tight')
            plt.clf()

        for i, x, y, u, d in zip(range(len(approved_x)), approved_x, approved_mean, approved_up, approved_down):
            plt.fill_between(x, u, d, color=color[i], alpha=0.2)
            plt.plot(x, y, color=color[i], linewidth=0.5, label="AC " + str(i))
        plt.xlim(0, 5000)  # set the xlim to xmin, xmax
        plt.ylim(0, 60)
        plt.legend()
        plt.savefig(base_dir + '/Traffic-A3C3-Comm' + str(comm) + '.pdf', bbox_inches='tight')
        plt.clf()
        approved_mean, approved_up, approved_down, approved_x = [], [], [], []
exit()"""
###############################################

# Navigation Rod
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials_for_acs = {0: {0: 0, 1: 0},
                           5: {0: 0, 1: 0, 2: 1, 3: 1},
                           10: {0: 0, 1: 0, 2: 1, 3: 1},
                           20: {0: 0, 1: 0, 2: 1, 3: 1}}
for server in ["TB-Rod-dcomm6"]:
    for comm in [0, 5, 10, 20]:
        for ac in range(4):
            for trial in range(2):
                xs, ys = [], []
                for train in range(trains):
                    dir = base_dir + server + '/Navigation-' + str(comm) + 'Comm-Trial' + str(trial) + \
                          '-AC' + str(ac) + '/train_' + str(train)
                    if not os.path.isdir(dir):
                        print(dir + " not found!")
                        continue

                    print("Reading", dir)
                    event_acc = event_accumulator.EventAccumulator(dir)
                    event_acc.Reload()

                    w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                    ghghgh =[299970, 299965, 299880]
                    if comm==20 and ac==1:
                        vals = vals[step_nums.index(ghghgh[train])+1:]
                        step_nums = step_nums[step_nums.index(ghghgh[train])+1:]
                    xs.append(step_nums)
                    ys.append(vals)

                if len(xs) == 0:
                    print("No trial found")
                    continue
                x, y = merge(xs, ys)
                mean_x = mean(x, avg_value)
                mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
                upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
                plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
                plt.plot(mean_x, mean_y, color=color[trial], label="Trial " + str(trial))

                if trial == approved_trials_for_acs[comm][ac]:
                    approved_mean.append(mean_y)
                    approved_up.append(upper_std)
                    approved_down.append(lower_std)
                    approved_x.append(mean_x)

            # plt.show()
            plt.xlim(0, 300000)  # set the xlim to xmin, xmax
            plt.ylim(0, 2)
            plt.legend()
            plt.savefig(base_dir + '/Navigation-A3C3-' + str(comm) + 'Comm-AC' + str(ac) + '.pdf',
                        bbox_inches='tight')
            plt.clf()

        for i, x, y, u, d in zip(range(len(approved_x)), approved_x, approved_mean, approved_up, approved_down):
            plt.fill_between(x, u, d, color=color[i], alpha=0.2)
            plt.plot(x, y, color=color[i], linewidth=0.5, label="AC " + str(i))
        plt.xlim(0, 300000)  # set the xlim to xmin, xmax
        plt.ylim(0, 2)
        plt.legend()
        plt.savefig(base_dir + '/Navigation-A3C3-Comm' + str(comm) + '.pdf', bbox_inches='tight')
        plt.clf()
        approved_mean, approved_up, approved_down, approved_x = [], [], [], []
exit()

###############################################

# Pursuit Retina
trains = 12
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials_for_acs = {0: {0: 0, 1: 0},
                           5: {0: 0, 1: 0, 2: 0, 3: 0},
                           10: {0: 0, 1: 0, 2: 0, 3: 0},
                           20: {0: 0, 1: 0, 2: 0, 3: 0}}
for server in ["TB-Retina-dcomm10.pursuit"]:
    for comm in [0, 5, 10, 20]:

        for ac in range(0, 4):
            for trial in range(1):
                xs, ys = [], []
                for train in range(trains):
                    dir = base_dir + server + '/Pursuit-' + str(comm) + 'Comm-Trial' + str(trial) + \
                          '-AC' + str(ac) + '/train_' + str(train)
                    if not os.path.isdir(dir):
                        print(dir + " not found!")
                        continue

                    print("Reading", dir)
                    event_acc = event_accumulator.EventAccumulator(dir)
                    event_acc.Reload()

                    w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                    xs.append(step_nums)
                    ys.append(vals)
                if len(xs) == 0:
                    print("No trial found")
                    continue
                x, y = merge(xs, ys)
                mean_x = mean(x, avg_value)
                mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
                upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
                plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
                plt.plot(mean_x, mean_y, color=color[trial], label="Trial " + str(trial))

                if trial == approved_trials_for_acs[comm][ac]:
                    approved_mean.append(mean_y)
                    approved_up.append(upper_std)
                    approved_down.append(lower_std)
                    approved_x.append(mean_x)

            # plt.show()
            plt.xlim(0, 200000)  # set the xlim to xmin, xmax
            plt.ylim(-1, 2)
            plt.legend()
            plt.savefig(base_dir + '/Pursuit-A3C3-' + str(comm) + 'Comm-AC' + str(ac) + '.pdf',
                        bbox_inches='tight')
            plt.clf()

        for i, x, y, u, d in zip(range(len(approved_x)), approved_x, approved_mean, approved_up, approved_down):
            plt.fill_between(x, u, d, color=color[i], alpha=0.2)
            plt.plot(x, y, color=color[i], linewidth=0.5, label="AC " + str(i))
        plt.xlim(0, 200000)  # set the xlim to xmin, xmax
        plt.ylim(-1, 2)
        plt.legend()
        plt.savefig(base_dir + '/Pursuit-A3C3-Comm' + str(comm) + '.pdf', bbox_inches='tight')
        plt.clf()
        approved_mean, approved_up, approved_down, approved_x = [], [], [], []

