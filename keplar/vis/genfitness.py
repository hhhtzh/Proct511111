import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save(name='tmp', h=None):
    stype = "pdf"
    name = name.strip().replace(' ', '-').replace('%', 'pct')
    if h == None:
        h = plt.gcf()
    h.tight_layout()
    print('saving', name + f'.{stype}')
    plt.savefig(name + f'.{stype}', bbox_inches='tight')
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
        while len(arr1) < 256:
            arr1.append(float(last))
        line = file.readline()
        mat.append(arr1)
    return mat


mat = readfile(open("/home/tzh/PycharmProjects/pythonProjectAR5/result/pmlb_1027_keplaroperon_generation.txt"))
mat = np.array(mat)
meancol = mat.mean(axis=0)
print(meancol)
arr2 = readfile(open("/home/tzh/PycharmProjects/pythonProjectAR5/result/pmlb_1027_keplarbingo_generation_fit.txt"))
arr2 = np.array(arr2)
meancol2 = arr2.mean(axis=0)
arr3 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_test8.json")
rl_arr = []
for i in range(99):
    tk = "data_gen" + str(i + 1)
    rl_arr.append(arr3[tk]["history_best_fit"])
rl_arr=np.array(rl_arr)


x = np.arange(0, 99, 1)

plt.plot(x, meancol2[0:99:1], label='KeplarBingo', color='red')
plt.plot(x, meancol[0:99:1], label='KeplarOperon', color='blue')
plt.plot(x,rl_arr[0:99:1],label='Keplar_PPO_RL', color='green')
plt.legend(loc='best')
plt.xlabel('Generation')
plt.ylabel('TrainFitness(RMSE)')
plt.savefig("/home/tzh/PycharmProjects/pythonProjectAR5/IMG_COLOR/LOG/pmlb_1027_kb_ko_rl_genfit.pdf")
plt.show()
