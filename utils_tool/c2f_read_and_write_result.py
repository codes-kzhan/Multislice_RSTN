import os
import numpy as np


#txt_file_path = '/export/ccvl12b/data2/results/coarse2fine_testing_1e-5x10,20/FD0:ZJ4_1,ZJ4_1,ZJ4_1:e2682_0.5_0.5,10/results.txt'
txt_file_path = '/export/ccvl12b/data2/results/coarse2fine_testing_1e-5x10,20/multigpu_Z_FD0:XJ4_1,YJ4_1,ZJ4_1:e2682_0.5_0.5,10/results.txt'
content = open(txt_file_path).read().splitlines()

case_num = -1
rounds = -1
dsc = np.zeros((11,20),dtype = np.float)
for line in content:
    if 'Round 0, Average DSC' in line:
        break
    if 'Testcase' in line:
        case_num += 1
        rounds = -1
    if 'Round' in line:
        rounds += 1
        num = float(line[line.find('0.'):line.find(' .')])
        print('Testcase ', case_num, ' rounds ', rounds, ' dsc =', num)
        dsc[rounds,case_num] = num

for i in range(11):
    print('round ', i, ' mean DSC = ',np.mean(dsc[i,:]))
