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


def plot_each_comm(trains, base_dir, avg_value, color, trials, foldernames, comms, environment, xlim, ylim):
    for server in foldernames:
        for comm in comms:
            for trial in trials:
                xs, ys = [], []
                for train in range(trains):
                    dir0 = base_dir + server + '/' + environment + '-' + str(comm) + 'Comm-Trial' + str(trial) + \
                           '/train_' + str(train)
                    if not os.path.isdir(dir0):
                        print(dir0 + " not found!")
                        continue

                    print("Reading", dir0)
                    event_acc = event_accumulator.EventAccumulator(dir0)
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
                plt.plot(mean_x, mean_y, color=color[trial])

            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])  # set the xlim to xmin, xmax
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            plt.savefig(base_dir + '/' + environment + '-A3C2-Comm' + str(comm) + '.pdf', bbox_inches='tight')
            plt.clf()


def plot_by_comm(trains, base_dir, avg_value, color, foldernames, environment, xlim, ylim, approved_trials):
    approved_mean, approved_up, approved_down, approved_x = [], [], [], []

    for server in foldernames:
        for comm in approved_trials.keys():
            trial = approved_trials[comm]

            xs, ys = [], []
            for train in range(trains):
                dir0 = base_dir + server + '/' + environment + '-' + str(comm) + 'Comm-Trial' + str(trial) + \
                       '/train_' + str(train)
                if not os.path.isdir(dir0):
                    print(dir0 + " not found!")
                    continue

                print("Reading", dir0)
                event_acc = event_accumulator.EventAccumulator(dir0)
                event_acc.Reload()

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            if len(xs) == 0:
                print("No trial found")
                continue
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25 if environment == "Pursuit" else 0.5 if environment == "BlindGroupUp" else 1)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            approved_mean.append(mean_y)
            approved_up.append(upper_std)
            approved_down.append(lower_std)
            approved_x.append(mean_x)

        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])  # set the xlim to xmin, xmax
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

    for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
        plt.fill_between(x, u, d, color=color[i], alpha=0.2)
        plt.plot(x, y, color=color[i],
                 label="No Comms" if i == 0 else str(list(approved_trials.keys())[i]) + " Channels")
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])  # set the xlim to xmin, xmax
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.savefig(base_dir + "/" + environment + '-A3C2.pdf', bbox_inches='tight')
    plt.clf()


trains = 3
avg_value = 100
base_dir = "/home/david/GitHub/DeepComm/"
color = ['red', 'blue', 'purple', 'yellow', 'green', 'pink']
color_comp = ['red', 'blue', 'purple', 'green']

###############################################

# plot_by_comm(3, base_dir, 10, color, ["."], "Traffic", [0, 1000], [0, 120], {0: 0, 1: 0, 2: 0, 5: 0})

# plot_each_comm(3, base_dir, 100, color, [0,1,2], ["Trials/TB-Rod-dcomm3.nav"], [0, 1, 5, 10, 20], "Navigation",
#             [0, 300000], [0,2])

# plot_by_comm(3, base_dir, 100, color, ["Trials/TB-Rod-dcomm3.nav"], "Navigation",
#             [0, 300000], [1,2], {0: 1, 1: 2, 5: 1, 10: 1, 20: 2})

# plot_by_comm(3, base_dir, 100, color, ["."], "Navigation",
#             [0, 200000], [0,2.01], {0: 0, 1: 0, 5: 0, 10: 0, 20: 0})

# plot_by_comm(3, base_dir, 10, color, ["Trials/A3C2"], "Traffic", [0, 1000], [0, 120], {0: 0, 1: 0, 2: 0, 5: 0})

#plot_by_comm(3, base_dir, 200, color, ["Trials/TB-Rod-dcomm7.bgu"], "BlindGroupUp",
#                          [0, 180000], [20, 100], {0: 1, 1: 2, 5: 2, 10: 1, 20: 1})
#plot_by_comm(3, base_dir, 200, color, ["Trials/A3C2"], "BlindGroupUp",
#             [0, 200000], [00, 100], {0: 0, 1: 0, 5: 0, 10: 0, 20: 0})

plot_by_comm(12, base_dir, 1000, color, ["Trials/TB-Rod-dcomm5.pursuit"], "Pursuit",
             [0, 170000], [-3,0], {0: 1, 1: 1, 5: 2, 10: 2, 20: 1})
