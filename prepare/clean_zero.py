import csv
import random

with open('../workload/join0/join0.csv', 'r') as f1, open('../workload/join0/join0.sparql', 'r') as f2:
    line1 = f1.readlines()
    line2 = f2.readlines()

    ls = list()

    for i in range(len(line1)):
        if int(line1[i].split(',')[-1]) > 0:
            ls.append([line1[i], line2[i]])

    # random.shuffle(ls)

    ls = ls[:1000]

    with open('../workload/join0/join01.csv', 'w') as f3, open('../workload/join0/join01.sparql', 'w') as f4:
        for i in range(len(ls)):
            f3.write(ls[i][0])
            f4.write(ls[i][1])

