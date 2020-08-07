import re
import numpy as np
from pprint import pprint


FILE = 'res_matrix_repeated.txt'
DUPLICATES = 10

def get_res(filename):
    with open(filename) as file:
        res = {}
        # sort data into dictionary
        for line in file: 
            pair =  re.findall(r'\w+-\w+', line)[0]
            accuracy = float(re.findall(r'[0-9]\.[0-9]*', line)[1])
            if pair in res:
                res[pair] += accuracy              
            else:
                res.update({pair: accuracy})

        # calculate accuracy average
        res_average = {k: v/DUPLICATES for k, v in res.items()}
        return res_average

def make_matrix(res_accuracy):

    M = np.zeros((20, 20))
    increasing_order = [None] * 19 # for interation of target matrix
    decreasing_order = [None] * 19 # for iteration of res_accuracy
    for i in range(1,20):
        increasing_order[i-1] = i
    for i in range(0,19):
        decreasing_order[i] = 18 - i
    
    label = [None] * 19
    count = 0
    for i in range(0,19):
        label[i] = count
        count += decreasing_order[i]+1
    label.append(190)
    
    for i in range(0,19):
        M[i][increasing_order[i]:] = res_accuracy[label[i]: label[i+1]]
        M.T[i][increasing_order[i]:] = res_accuracy[label[i]: label[i+1]] 
        print('end of %d loop' % i)
    
    return M

def save_matrix_csv(matrix):
    np.savetxt('res_matrix.csv', matrix, delimiter=',')

if __name__ == '__main__':
    res = get_res(FILE)
    accuracy_list = [res[x] for x in sorted(res)]
    
    res_matrix = make_matrix(accuracy_list)
    save_matrix_csv(res_matrix)
    pprint(res_matrix)
