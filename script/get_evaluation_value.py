import numpy as np
import pandas as pd
import argparse
import os


def get_avg(csv_path, attr_index):
    f = open(csv_path, "r")
    lines = f.readlines()
    avg = 0.
    count = 0.
    for idx, line in enumerate(lines):
        if idx > 0:
            avg += float(line.split(',')[attr_index])
            count += 1

    return avg/count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="exp_s2_edge")
    parser.add_argument('--attr', type=int, default=3)
    args = parser.parse_args()

    exp_dir = args.exp
    attr_index = args.attr
    attr_max = .0
    attr_max_epoch_dir = ''
    for dir in sorted(os.listdir(exp_dir)):
        dir_path = os.path.join(exp_dir, dir)
        if os.path.isdir(dir_path):
            print(dir_path)
            attr_avg = get_avg(os.path.join(dir_path, 'val.csv'), attr_index)
            print(attr_avg)

            if attr_avg > attr_max:
                attr_max = attr_avg
                attr_max_epoch_dir = dir_path

    print("attr index = {}, max = {} @ {}".format(attr_index, attr_max, attr_max_epoch_dir))
