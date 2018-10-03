import os
import numpy as np


#txt_file_path = "/export/ccvl12b/data2/results/coarse_testing_1e-5x10,20/FD0:ZJ4_1.pkl/results.txt"
txt_file_path = "/export/ccvl12b/data2/results/coarse_testing_1e-5x10,20/FD0:YJ4_1.pkl/results.txt"
txt_file_path = "/export/ccvl12b/data2/results/coarse_testing_1e-5x10,20/FD0:XJ4_1.pkl/results.txt"

content = open(txt_file_path).read().splitlines()

case_num = -1
dsc = np.zeros((20),dtype = np.float)
for line in content:
    if 'average' in line:
        break
    if 'Testcase' in line:
        case_num += 1
        num = float(line[line.find('0.'):line.find(' .')])
        print('Testcase ', case_num,' dsc =', num)
        dsc[case_num] = num


print('mean DSC = ',np.mean(dsc[:]))
