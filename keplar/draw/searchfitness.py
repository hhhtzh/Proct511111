import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save(name='tmp',h=None):
    stype = "pdf"
    name = name.strip().replace(' ','-').replace('%','pct')
    if h == None:
        h = plt.gcf()
    h.tight_layout()
    print('saving',name+f'.{stype}')
    plt.savefig(name+f'.{stype}', bbox_inches='tight')
    # plt.savefig(name + '.png')


def readfile(file):
    mat = []
    line = file.readline()
    while line:
        arr1 = []
        while line != "\n":
            data = line.split()
            print(data)
            if len(data) > 0:
                arr1.append(float(data[1]))
                last = data[1]
                line = file.readline()
            else:
                break
        while len(arr1) < 10000:
            arr1.append(float(last))
        line = file.readline()
        mat.append(arr1)
    return mat


mat = readfile(open("/home/tzh/PycharmProjects/pythonProjectAR5/result/pmlb_1027_keplaroperon_searchfit.txt"))
mat = np.array(mat)
meancol = mat.mean(axis=0)

arr2 = readfile(open("/home/tzh/PycharmProjects/pythonProjectAR5/result/pmlb_1027_keplarbingo_searchfit.txt"))
arr2 = np.array(arr2)
meancol2 = arr2.mean(axis=0)

x = np.arange(0, 10000, 1)

plt.plot(x, meancol2[0:10000:1], label='KeplarBingo', color='red')
plt.plot(x, meancol[0:10000:1], label='KeplarOperon', color='blue')
plt.legend(loc='best')
plt.xlabel('SearchNum')
plt.ylabel('Fitness')
plt.savefig("/home/tzh/PycharmProjects/pythonProjectAR5/IMG_COLOR/LOG/pmlb_1027_kb_ko_searchfit.pdf")
plt.show()

