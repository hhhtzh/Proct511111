import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def readfile(file):
    mat=[]
    line = file.readline()
    while line:
        arr1 = []
        while line!="\n" :
            data = line.split()
            arr1.append(float(data[1]))
            last=data[1]
            line = file.readline()
        while len(arr1)<10000:
            arr1.append(float(last))
        line = file.readline()
        mat.append(arr1)
    return mat

mat=readfile(open("gplearn/kornstwo2.txt"))
mat=np.array(mat)
meancol=mat.mean(axis=0)

# arr2=readfile(open("gplearn_c/kornstwo2.txt"))
# arr2=np.array(arr2)
# meancol2=arr2.mean(axis=0)

x = np.arange(0, 10000, 1)

# plt.plot(x,meancol2[0:10000:500], label='gp', color='red')
plt.plot(x,meancol[0:10000:1], label='steiner', color='blue')
plt.legend(loc='best')
plt.xlabel('Generation')
plt.ylabel('Fitness')


plt.show()