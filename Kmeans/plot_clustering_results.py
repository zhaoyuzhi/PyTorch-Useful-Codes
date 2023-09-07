import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# read a txt expect EOF
def text_readlines(filename):
    # try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type = str, default = 'report.txt', help = 'report')
    opt = parser.parse_args()

    # build report
    report = text_readlines(opt.report)
    
    # build data list
    k = np.arange(50, 2101, 50)
    sum_of_distance_list = []
    top1_accuracy_list = []
    sum_of_distance_template = 'Sum of distances:'
    top1_accuracy_template = 'Top 1 accuracy:'

    for i in range(len(k)):
        for content in report:
            k_value_str = 'K value: %d' % k[i]
            if k_value_str in content:
                if sum_of_distance_template in content:
                    temp = float(content.split(sum_of_distance_template)[1].split('|').strip())
                    sum_of_distance_list.append(temp)
                if top1_accuracy_template in content:
                    temp = float(content.split(top1_accuracy_template)[1].strip())
                    top1_accuracy_list.append(temp)

    # plot
    plt.plot(k, sum_of_distance_list)
    plt.show()
    