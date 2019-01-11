import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np


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

# Blind Group Up Retina
"""approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0:0, 1:0, 5:0, 10:2, 20:1}
for server in ["TB-Retina-dcomm7"]:
    for comm in [0,1,5,10,20]:
        for trial in range(3):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/BlindGroupUp-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        #plt.show()
        plt.xlim(0, 200000)  # set the xlim to xmin, xmax
        plt.ylim(10, 100)
        plt.savefig(base_dir+'/BlindGroupUp-A3C3-'+str(comm)+'Comm.pdf', bbox_inches='tight')
        plt.clf()

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i], linewidth=0.5)
plt.xlim(0, 200000)  # set the xlim to xmin, xmax
plt.ylim(10, 100)
plt.savefig(base_dir+'/BlindGroupUp-A3C3.pdf', bbox_inches='tight')
plt.clf()

# Blind Group Up Rod, c5t2 c10t2 c20t2 fake
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0:1, 1:2, 5:2, 10:1, 20:1}
for server in ["TB-Rod-dcomm7"]:
    for comm in [0,1,5,10,20]:
        for trial in range(3):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/BlindGroupUp-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        #plt.show()
        plt.xlim(0, 200000)  # set the xlim to xmin, xmax
        plt.ylim(10, 100)
        plt.savefig(base_dir+'/BlindGroupUp-A3C2-'+str(comm)+'Comm.pdf', bbox_inches='tight')
        plt.clf()

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i], linewidth=0.5)
plt.xlim(0, 200000)  # set the xlim to xmin, xmax
plt.ylim(10, 100)
plt.savefig(base_dir + '/BlindGroupUp-A3C2.pdf', bbox_inches='tight')
plt.clf()"""

########################################################
# compare BGU

# Blind Group Up Retina
"""approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0:0, 20:1}
for server in ["TB-Retina-dcomm7"]:
    for comm in [0,20]:
        for trial in [0,1]:
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/BlindGroupUp-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

# Blind Group Up Rod,
approved_trials = {0:1,20:1}
for server in ["TB-Rod-dcomm7"]:
    for comm in [0,20]:
        for trial in [0,1]:
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/BlindGroupUp-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.25)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

for i, x, y, u, d in zip(range(4), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color_comp[i], alpha=0.2)
    plt.plot(x, y, color=color_comp[i], linewidth=0.5)
plt.xlim(0, 200000)  # set the xlim to xmin, xmax
plt.ylim(10, 100)
plt.savefig(base_dir + '/BlindGroupUp.pdf', bbox_inches='tight')
plt.clf()
exit()"""

###########################################

"""
avg_value = 50

# Traffic Retina
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 1, 1: 2, 2: 0, 5: 0}
for server in ["TB-Retina-dcomm8"]:
    for comm in [0,1,2,5]:
        for trial in range(3):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Traffic-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, 0.5)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        #plt.show()
        plt.xlim(0, 20000)  # set the xlim to xmin, xmax
        plt.ylim(0, 55)
        plt.savefig(base_dir+'/Traffic-A3C3-'+str(comm)+'Comm.pdf', bbox_inches='tight')
        plt.clf()

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i])
plt.xlim(0, 20000)    # set the xlim to xmin, xmax
plt.ylim(0, 55)
plt.savefig(base_dir + '/Traffic-A3C3.pdf', bbox_inches='tight')
plt.clf()

approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 0, 1: 1, 2: 0, 5: 0}

# Traffic Rod
for server in ["TB-Rod-dcomm8"]:
    for comm in [0,1,2,5]:
        for trial in range(2):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Traffic-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, 0.5)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        #plt.show()
        plt.xlim(0, 20000)  # set the xlim to xmin, xmax
        plt.ylim(0, 55)
        plt.savefig(base_dir+'/Traffic-A3C2-'+str(comm)+'Comm.pdf', bbox_inches='tight')
        plt.clf()

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i])
plt.xlim(0, 20000)    # set the xlim to xmin, xmax
plt.ylim(0, 55)
plt.savefig(base_dir + '/Traffic-A3C2.pdf', bbox_inches='tight')
plt.clf()


###########################################
# Traffic comp

# Traffic Retina
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 1, 5: 0}
for server in ["TB-Retina-dcomm8"]:
    for comm in [0,5]:
        for trial in range(2):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Traffic-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, 0.5)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

# Traffic Rod,
approved_trials = {0: 0, 5: 0}
for server in ["TB-Rod-dcomm8"]:
    for comm in [0,5]:
        for trial in range(1):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Traffic-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, 0.5)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color_comp[i], alpha=0.2)
    plt.plot(x, y, color=color_comp[i])
plt.xlim(0, 20000)    # set the xlim to xmin, xmax
plt.ylim(0, 55)
plt.savefig(base_dir + '/Traffic.pdf', bbox_inches='tight')
plt.clf()
"""

################################

# Pursuit Retina
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 1, 1: 1, 5: 2, 10: 1, 20: 1}
for server in ["TB-Retina-dcomm5"]:
    for comm in [0, 1, 5, 10, 20]:
        for trial in [1, 2]:
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir + server + '/Pursuit-' + str(comm) + 'Comm-Trial' + str(trial) + '/train_' + str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.3)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        # plt.show()
        plt.xlim(0, 300000)  # set the xlim to xmin, xmax
        plt.ylim(-3, 0)
        plt.savefig(base_dir + '/Pursuit-A3C3-' + str(comm) + 'Comm.pdf', bbox_inches='tight')
        plt.clf()
for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i])
plt.xlim(0, 300000)  # set the xlim to xmin, xmax
plt.ylim(-3, 0)
plt.savefig(base_dir + '/Pursuit-A3C3.pdf', bbox_inches='tight')
plt.clf()

# Pursuit Rod,
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 1, 1: 1, 5: 2, 10: 2, 20: 1}
for server in ["TB-Rod-dcomm5"]:
    for comm in [0, 1, 5, 10, 20]:
        for trial in [1, 2]:
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir + server + '/Pursuit-' + str(comm) + 'Comm-Trial' + str(trial) + '/train_' + str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.3)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        # plt.show()
        plt.xlim(0, 300000)  # set the xlim to xmin, xmax
        plt.ylim(-3, 0)
        plt.savefig(base_dir + '/Pursuit-A3C2-' + str(comm) + 'Comm.pdf', bbox_inches='tight')
        plt.clf()

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i])
plt.xlim(0, 300000)  # set the xlim to xmin, xmax
plt.ylim(-3, 0)
plt.savefig(base_dir + '/Pursuit-A3C2.pdf', bbox_inches='tight')
plt.clf()

########


# Pursuit Retina
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 1, 10: 1}
for server in ["TB-Retina-dcomm5"]:
    for comm in [0, 10]:
        for trial in [1]:
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir + server + '/Pursuit-' + str(comm) + 'Comm-Trial' + str(trial) + '/train_' + str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.3)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

# Pursuit Rod,
approved_trials = {0: 1, 20: 1}
for server in ["TB-Rod-dcomm5"]:
    for comm in [0, 20]:
        for trial in [1]:
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir + server + '/Pursuit-' + str(comm) + 'Comm-Trial' + str(trial) + '/train_' + str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, std_multiplier=0.3)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

for i, x, y, u, d in zip(range(4), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color_comp[i], alpha=0.2)
    plt.plot(x, y, color=color_comp[i])
plt.xlim(0, 300000)  # set the xlim to xmin, xmax
plt.ylim(-3, 0)
plt.savefig(base_dir + '/Pursuit.pdf', bbox_inches='tight')
plt.clf()

###########################################


# Pursuit Retina LENGTH
"""approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 0, 1: 0, 5: 0, 10: 0}
for server in ["TB-Retina-dcomm4"]:
    for comm in [0,1,5, 10]:
        for trial in range(1):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Pursuit-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Length'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        #plt.show()
        plt.xlim(0, 100000)  # set the xlim to xmin, xmax
        plt.ylim(30, 80)
        plt.savefig(base_dir+'/Pursuit-A3C3-'+str(comm)+'Comm.pdf', bbox_inches='tight')
        plt.clf()
for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i])
plt.xlim(0, 100000)  # set the xlim to xmin, xmax
plt.ylim(30, 80)
plt.savefig(base_dir + '/Pursuit-A3C3.pdf', bbox_inches='tight')
plt.clf()

# Pursuit Rod,
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 0, 1: 0, 5: 0}
for server in ["TB-Rod-dcomm4"]:
    for comm in [0,1,5]:
        for trial in range(1):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Pursuit-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Length'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        #plt.show()
        plt.xlim(0, 100000)  # set the xlim to xmin, xmax
        plt.ylim(30, 80)
        plt.savefig(base_dir+'/Pursuit-A3C2-'+str(comm)+'Comm.pdf', bbox_inches='tight')
        plt.clf()

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i])
plt.xlim(0, 100000)  # set the xlim to xmin, xmax
plt.ylim(30, 80)
plt.savefig(base_dir + '/Pursuit-A3C2.pdf', bbox_inches='tight')
plt.clf()"""

########


# Pursuit Retina LENGTH COMP
"""approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 0, 10: 0}
for server in ["TB-Retina-dcomm4"]:
    for comm in [0,10]:
        for trial in range(1):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Pursuit-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Length'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)


# Pursuit Rod,
approved_trials = {0: 0, 1: 0}
for server in ["TB-Rod-dcomm4"]:
    for comm in [0,1]:
        for trial in range(1):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Pursuit-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Length'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

for i, x, y, u, d in zip(range(4), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color_comp[i], alpha=0.2)
    plt.plot(x, y, color=color_comp[i])
plt.xlim(0, 100000)  # set the xlim to xmin, xmax
plt.ylim(30, 80)
plt.savefig(base_dir + '/Pursuit.pdf', bbox_inches='tight')
plt.clf()"""

###########################################

"""avg_value = 50

# Nav Retina
approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 2, 1: 3, 5: 1, 10: 1, 20: 2}
for server in ["TB-Retina-dcomm3"]:
    for comm in [0,1,5,10,20]:
        for trial in range(4):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Navigation-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, 0.5)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        #plt.show()
        plt.xlim(0, 300000)  # set the xlim to xmin, xmax
        plt.ylim(1, 2)
        plt.savefig(base_dir+'/Navigation-A3C3-'+str(comm)+'Comm.pdf', bbox_inches='tight')
        plt.clf()

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i])
plt.xlim(0, 300000)    # set the xlim to xmin, xmax
plt.ylim(1, 2)
plt.savefig(base_dir + '/Navigation-A3C3.pdf', bbox_inches='tight')
plt.clf()

approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 1, 1: 2, 5: 1, 10: 1, 20: 2}

# Navigation Rod
for server in ["TB-Rod-dcomm3"]:
    for comm in [0,1,5,10,20]:
        for trial in range(3):
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Navigation-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, 0.5)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)
            plt.fill_between(mean_x, upper_std, lower_std, color=color[trial], alpha=0.2)
            plt.plot(mean_x, mean_y, color=color[trial])

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)
        #plt.show()
        plt.xlim(0, 300000)  # set the xlim to xmin, xmax
        plt.ylim(1, 2)
        plt.savefig(base_dir+'/Navigation-A3C2-'+str(comm)+'Comm.pdf', bbox_inches='tight')
        plt.clf()

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color[i], alpha=0.2)
    plt.plot(x, y, color=color[i])
plt.xlim(0, 300000)    # set the xlim to xmin, xmax
plt.ylim(1, 2)
plt.savefig(base_dir + '/Navigation-A3C2.pdf', bbox_inches='tight')
plt.clf()
exit()"""

###########################################
# Navigation comp

# Navigation Retina
"""approved_mean, approved_up, approved_down, approved_x = [], [], [], []
approved_trials = {0: 2, 20: 2}
for server in ["TB-Retina-dcomm3"]:
    for comm in [0,20]:
        for trial in [2]:
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Navigation-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, 0.5)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

# Navigation Rod,
approved_trials = {0: 1, 5: 1}
for server in ["TB-Rod-dcomm3"]:
    for comm in [0,5]:
        for trial in [1]:
            xs, ys = [], []
            for train in range(trains):
                dir = base_dir+server+'/Navigation-'+str(comm)+'Comm-Trial'+str(trial)+'/train_'+str(train)
                print("Reading", dir)
                event_acc = event_accumulator.EventAccumulator(dir)
                event_acc.Reload()

                # Show all tags in the log file
                # print(event_acc.Tags())

                w_times, step_nums, vals = zip(*event_acc.Scalars('Perf/Reward'))
                xs.append(step_nums)
                ys.append(vals)
            x, y = merge(xs, ys)
            mean_x = mean(x, avg_value)
            mean_y, std_y = mean_std(y, avg_value, 0.5)
            upper_std, lower_std = np.sum([mean_y, std_y], axis=0), np.subtract(mean_y, std_y)

            if trial == approved_trials[comm]:
                approved_mean.append(mean_y)
                approved_up.append(upper_std)
                approved_down.append(lower_std)
                approved_x.append(mean_x)

for i, x, y, u, d in zip(range(5), approved_x, approved_mean, approved_up, approved_down):
    plt.fill_between(x, u, d, color=color_comp[i], alpha=0.2)
    plt.plot(x, y, color=color_comp[i])
plt.xlim(0, 200000)    # set the xlim to xmin, xmax
plt.ylim(1, 2)
plt.savefig(base_dir + '/Navigation.pdf', bbox_inches='tight')
plt.clf()"""
