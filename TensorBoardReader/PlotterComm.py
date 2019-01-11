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


def plot_by_trial(trains, base_dir, avg_value, color, acs, trials, foldernames, comms, environment, xlim, ylim):
    trial_comm_name = ["Loss", "Noise", "Jumble", "All", "None"]

    for server in foldernames:
        for comm in comms:
            for trial in trials:
                for ac in acs:
                    xs, ys = [], []
                    for train in range(trains):
                        dir0 = base_dir + server + '/' + environment + '-' + str(comm) + 'Comm-AC' + str(ac) + \
                               '-Trial' + str(trial)
                        dir1 = base_dir + server + '/' + environment + '-' + str(comm) + 'Comm-Trial' + str(trial) + \
                               '-AC' + str(ac)
                        dir2 = dir0 + "-" + trial_comm_name[trial]
                        if os.path.isdir(dir0):
                            os.rename(dir0, dir2)
                        if os.path.isdir(dir1):
                            os.rename(dir1, dir2)
                        dir2 = dir2 + '/train_' + str(train)
                        if not os.path.isdir(dir2):
                            print(dir2 + " not found!")
                            continue

                        print("Reading", dir2)
                        event_acc = event_accumulator.EventAccumulator(dir2)
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
                    plt.fill_between(mean_x, upper_std, lower_std, color=color[ac], alpha=0.2)
                    plt.plot(mean_x, mean_y, color=color[ac], label="AC " + str(ac))

                if xlim is not None:
                    plt.xlim(xlim[0], xlim[1])  # set the xlim to xmin, xmax
                if ylim is not None:
                    plt.ylim(ylim[0], ylim[1])
                plt.legend()
                plt.savefig(base_dir + '/' + environment + '-CMA4C-Comm' + str(comm) + '-' +
                            trial_comm_name[trial] + '.pdf', bbox_inches='tight')
                plt.clf()


def plot_by_ac(trains, base_dir, avg_value, color, acs, trials, foldernames, comms, environment, xlim, ylim, baseline=None):
    trial_comm_name = ["Loss", "Noise", "Jumble", "All", "None"]

    for server in foldernames:
        for comm in comms:
            for ac in acs:
                for trial in trials:
                    xs, ys = [], []
                    for train in range(trains):
                        dir0 = base_dir + server + '/' + environment + '-' + str(comm) + 'Comm-AC' + str(ac) + \
                               '-Trial' + str(trial)
                        dir1 = base_dir + server + '/' + environment + '-' + str(comm) + 'Comm-Trial' + str(trial) + \
                               '-AC' + str(ac)
                        dir2 = dir0 + "-" + trial_comm_name[trial]
                        if os.path.isdir(dir0):
                            os.rename(dir0, dir2)
                        if os.path.isdir(dir1):
                            os.rename(dir1, dir2)
                        dir2 = dir2 + '/train_' + str(train)
                        if not os.path.isdir(dir2):
                            print(dir2 + " not found!")
                            continue

                        print("Reading", dir2)
                        event_acc = event_accumulator.EventAccumulator(dir2)
                        event_acc.Reload()

                        w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                        if trial == 3 and environment is "Navigation":
                            # print(step_nums)
                            xs.append(step_nums[step_nums.index([50, 40, 299955][train]) + 2:])
                            ys.append(vals[step_nums.index([50, 40, 299955][train]) + 2:])
                        else:
                            xs.append(step_nums)
                            ys.append(vals)
                    if len(xs) == 0:
                        print("No trial found")
                        continue
                    x, y = merge(xs, ys)
                    mean_x = mean(x, avg_value)
                    mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25 if environment=="Pursuit" else 0.5 if environment == "BlindGroupUp" else 1)
                    upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
                    plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
                    plt.plot(mean_x, mean_y, color=color[trial], label=trial_comm_name[trial])

                if baseline is not None:
                    plt.plot([xlim[0], xlim[1]], [baseline,baseline], '--', color="black", label="No Noise")

                if xlim is not None:
                    plt.xlim(xlim[0], xlim[1])  # set the xlim to xmin, xmax
                if ylim is not None:
                    plt.ylim(ylim[0], ylim[1])
                plt.legend()
                plt.savefig(base_dir + '/' + environment + '-A3C2-Comm' + str(comm) + '-AC' +
                            str(ac) + '.pdf', bbox_inches='tight')
                plt.clf()


trains = 3
avg_value = 100
base_dir = "/home/david/GitHub/DeepComm/"
color = ['red', 'blue', 'purple', 'yellow', 'green', 'pink']
color_comp = ['red', 'blue', 'purple', 'green']

###############################################

plot_by_ac(3, base_dir, 200, color, [1], [0,1,2,3], ["Trials/A3C2"], [20], "BlindGroupUp",
           [0, 180000], [20, 100], baseline=78)
exit()

# plot_by_trial(3, base_dir, 10, color, [0,1,2,3], [0,1,2,3,4], ["Lily.traffic.comm"], [5], "Traffic", [0, 2000], [0, 110])
# plot_by_ac(3, base_dir, 2, color, [0,1,2,3], [0,1,2,3,4], ["Lily.traffic.comm"], [5], "Traffic", [0, 500], [0, 110])
plot_by_ac(3, base_dir, 10, color, [0], [0,1,2,3], ["Trials/A3C2"], [5], "Traffic",
           [0, 1000], [0, 120], baseline=99)

# plot_by_trial(3, base_dir, 10, color, [3], [0, 1, 2, 3, 4], ["TB-Rod-dcomm14.nav.comm"], [20], "Navigation", [0, 200000], [0, 2.1])
# plot_by_ac(3, base_dir, 10, color, [3], [0, 1, 2, 3, 4], ["TB-Rod-dcomm14.nav.comm"], [20], "Navigation", [0, 200000], [0, 2.1])
plot_by_ac(3, base_dir, 100, color, [0], [0, 1, 3], ["Trials/A3C2"], [20], "Navigation",
           [0, 200000], [0, 2.01], baseline = 2)

# plot_by_trial(12, base_dir, avg_value, color, [0,1,2,3], [0,1], ["TB-Retina-dcomm13.pursuit.comm"], [10], "Pursuit", [0, 200000], [-10, 0])
#plot_by_ac(12, base_dir, avg_value, color, [0, 1, 2, 3], [0, 1], ["TB-Retina-dcomm13.pursuit.comm"], [10], "Pursuit",
#           [0, 200000], [-10, 0])
plot_by_ac(12, base_dir, 1000, color, [0], [0, 1, 2, 3], ["TB-Retina-dcomm15"], [10], "Pursuit",
           [0, 170000], [-6, 0], baseline=-1.1)

# plot_by_trial(3, base_dir, avg_value, color, [0,1,2,3], [0,1], ["TB-Rod-dcomm12.bgu.comm"], [20], "BlindGroupUp", [0, 200000], [25, 85])
#plot_by_ac(3, base_dir, 1000, color, [0, 1, 2, 3], [0, 1], ["TB-Rod-dcomm12.bgu.comm"], [20], "BlindGroupUp",
#           [0, 200000], [25, 85])
plot_by_ac(3, base_dir, 200, color, [0], [0,1,2,3], ["Trials/A3C2"], [20], "BlindGroupUp",
           [0, 200000], [0, 100], baseline=85)
