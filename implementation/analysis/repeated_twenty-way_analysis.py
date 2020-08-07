import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

FILE = 'res_baseline.txt'

def get_res(file):
    with open(file) as res_file:
        res_loss = []
        res_accuracy = []
        for line in res_file:
            res_loss.append(float(re.findall(r'[0-9]\.[0-9]*', line)[0]))
            res_accuracy.append(float(re.findall(r'[0-9]\.[0-9]*', line)[1]))
        return res_loss, res_accuracy

def cal():
    loss, accuracy = get_res(FILE)
    average_loss, std_loss = np.mean(loss), np.std(loss)
    average_accuracy, std_accuracy = np.mean(accuracy), np.std(accuracy)
    print("Loss-mean: {0} Loss-std: {1}".format(average_loss, std_loss))
    print("Accuracy-mean: {0} Accuracy-std: {1}".format(average_accuracy, std_accuracy))

if __name__ == '__main__':
    cal()