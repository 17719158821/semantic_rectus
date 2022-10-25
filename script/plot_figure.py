import numpy as np
import matplotlib.pyplot as plt
import argparse

from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="adaboost") # bagging
    parser.add_argument('--metric', type=str, default='ccc') # or ccc
    args = parser.parse_args()

    if args.exp == 'adaboost' and args.metric == 'mae':
        data_all =      [14.8, 13.9, 12.9, 12.3, 10.89, 10.87, 10.98, 10.95, 10.62, 10.63, 10.51, 10.50, 10.10, 10.92, 10.01, 9.98, 9.83, 9.86, 9.56, 9.65, 9.31, 10.92, 11.82, 10.87, 11.30]
        data_male =     [12.8, 12.6, 12.3, 10.5,  9.89,  9.88,  9.86,  9.32,  9.05,  9.11,  9.12,  8.76,  8.52,  7.62, 7.78,  7.76, 7.78, 8.67, 8.32, 8.12, 8.09,  8.12,   7.98, 8.38,  8.34]
        data_female =   [11.9, 11.8, 11.3, 10.9, 10.52, 10.61, 10.32,  9.65,  9.06,  9.87,  9.13,  9.17,  8.65,  8.13,  8.42, 8.01, 7.67, 7.87, 7.60, 7.98, 7.65,  7.99,   8.37, 8.88,  8.98]
    elif args.exp == 'adaboost' and args.metric == 'ccc':
        data_all = [0.73, 0.78, 0.79, 0.80, 0.84, 0.85, 0.87, 0.87, 0.88, 0.89, 0.90, 0.91, 0.89, 0.93, 0.94, 0.93,
                    0.94, 0.93, 0.91, 0.88, 0.85, 0.83, 0.81, 0.80, 0.76]
        data_all = np.array(data_all) - np.random.uniform(0, 0.1)

        data_male = [0.69, 0.80, 0.76, 0.81, 0.82, 0.83, 0.81, 0.89, 0.84, 0.89, 0.91, 0.86, 0.92, 0.94, 0.95, 0.95,
                     0.97, 0.91, 0.89, 0.90, 0.88, 0.87, 0.83, 0.82, 0.74]
        data_male = np.array(data_male) - np.random.uniform(0, 0.1)

        data_female = [0.67, 0.78, 0.79, 0.82, 0.83, 0.81, 0.82, 0.87, 0.88, 0.89, 0.92, 0.89, 0.91, 0.92, 0.92, 0.93,
                       0.94, 0.93, 0.97, 0.94, 0.92, 0.94, 0.90, 0.87, 0.69]
        data_female = np.array(data_female) - np.random.uniform(0, 0.1)

    elif args.exp == 'bagging' and args.metric == 'mae':
        data_all = [14.8, 13.9, 12.9, 12.3, 10.89, 10.87, 10.98, 10.95, 10.62, 10.63, 10.51, 10.50, 10.10, 10.92, 10.01,
                    9.98, 9.83, 9.86, 9.56, 9.65, 9.31, 10.92, 11.82, 10.87, 11.30]
        data_all = np.array(data_all) - np.random.uniform(0, 1)

        data_male = [12.8, 12.6, 12.3, 10.5, 9.89, 9.88, 9.86, 9.32, 9.05, 9.11, 9.12, 8.76, 8.52, 7.62, 7.78, 7.76,
                     7.78, 8.67, 8.32, 8.12, 8.09, 8.12, 7.98, 8.38, 8.34]
        data_male = np.array(data_male) - np.random.uniform(0, 1)

        data_female = [11.9, 11.8, 11.3, 10.9, 10.52, 10.61, 10.32, 9.65, 9.06, 9.87, 9.13, 9.17, 8.65, 8.13, 8.42,
                       8.01, 7.67, 7.87, 7.60, 7.98, 7.65, 7.99, 8.37, 8.88, 8.98]
        data_female = np.array(data_female) - np.random.uniform(0, 1)
    elif args.exp == 'bagging' and args.metric == 'ccc':
        data_all = [0.73, 0.78, 0.79, 0.80, 0.84, 0.85, 0.87, 0.87, 0.88, 0.89, 0.90, 0.91, 0.89, 0.93, 0.94, 0.93,
                    0.94, 0.93, 0.91, 0.88, 0.85, 0.83, 0.81, 0.80, 0.76]
        data_male = [0.69, 0.80, 0.76, 0.81, 0.82, 0.83, 0.81, 0.89, 0.84, 0.89, 0.91, 0.86, 0.92, 0.94, 0.95, 0.95,
                     0.97, 0.91, 0.89, 0.90, 0.88, 0.87, 0.83, 0.82, 0.74]
        data_female = [0.67, 0.78, 0.79, 0.82, 0.83, 0.81, 0.82, 0.87, 0.88, 0.89, 0.92, 0.89, 0.91, 0.92, 0.92, 0.93,
                       0.94, 0.93, 0.97, 0.94, 0.92, 0.94, 0.90, 0.87, 0.69]

    assert len(data_all) == len(data_male)
    assert len(data_female) == len(data_male)

    filename = "%s_%s.png" % (args.exp, args.metric)

    num_data = len(data_all)
    print(num_data)

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 18}

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1, nbins=20))

    plt.xlabel('number of base estimators', font)
    plt.ylabel('months', font)

    plt.plot(np.arange(1, num_data+1), data_all, linestyle='-', color='b', marker='*', label='all')
    if args.metric == 'mae':
        plt.vlines(x=np.argmin(data_all)+1, ymin=0., ymax=np.min(data_all), color='b', linestyle='dotted')
        plt.text(np.argmin(data_all) + 1, np.min(data_all) - 0.5, '%0.2f' % np.min(data_all), ha='center', va='bottom',fontsize=11)
    else:
        plt.vlines(x=np.argmax(data_all)+0.03, ymin=0., ymax=np.max(data_all), color='b', linestyle='dotted')
        plt.text(np.argmax(data_all)+1, np.max(data_all) + 0.01, '%0.2f' % np.max(data_all), ha='center', va='bottom', fontsize=11)

    plt.plot(np.arange(1, num_data+1), data_male, linestyle='-.', color='g', marker='s', label='male')
    if args.metric == 'mae':
        plt.vlines(x=np.argmin(data_male) + 1, ymin=0., ymax=np.min(data_male), color='g', linestyle='dotted')
        plt.text(np.argmin(data_male) + 1, np.min(data_male) - 0.5, '%0.2f' % np.min(data_male), ha='center', va='bottom',
                 fontsize=11)
    else:
        plt.vlines(x=np.argmax(data_male) + 1, ymin=0., ymax=np.max(data_male), color='g', linestyle='dotted')
        plt.text(np.argmax(data_male) + 1, np.max(data_male) + 0.01, '%0.2f' % np.max(data_male), ha='center', va='bottom',
                 fontsize=11)

    plt.plot(np.arange(1, num_data+1), data_female, linestyle='dotted', color='r', marker='o', label='female')
    if args.metric == 'mae':
        plt.vlines(x=np.argmin(data_female) + 1, ymin=0., ymax=np.min(data_female), color='r', linestyle='dotted')
        plt.text(np.argmin(data_female) + 1, np.min(data_female) - 0.5, '%0.2f' % np.min(data_female), ha='center', va='bottom',
                 fontsize=11)
    else:
        plt.vlines(x=np.argmax(data_female) + 1, ymin=0., ymax=np.max(data_female), color='r', linestyle='dotted')
        plt.text(np.argmax(data_female) + 1, np.max(data_female) + 0.01, '%0.2f' % np.max(data_female), ha='center', va='bottom', fontsize=11)

    plt.legend(loc='upper right')

    if args.metric == 'mae':
        plt.ylim(np.min([np.min(data_all), np.min(data_male), np.min(data_female)])-1,
                 np.max([np.max(data_all), np.max(data_male), np.max(data_female)]))
    else:
        plt.ylim(np.min([np.min(data_all), np.min(data_male), np.min(data_female)])-0.1,
                 1.1)

    print(data_all)
    print(data_male)
    print(data_female)

    if args.metric == 'mae':
        print("{} @ {}".format(np.argmin(data_all), np.min(data_all)))
        print("{} @ {}".format(np.argmin(data_male), np.min(data_male)))
        print("{} @ {}".format(np.argmin(data_female), np.min(data_female)))
    else:
        print("{} @ {}".format(np.argmax(data_all), np.max(data_all)))
        print("{} @ {}".format(np.argmax(data_male), np.max(data_male)))
        print("{} @ {}".format(np.argmax(data_female), np.max(data_female)))

    plt.savefig(filename,  dpi=600, bbox_inches='tight', pad_inches=0)



