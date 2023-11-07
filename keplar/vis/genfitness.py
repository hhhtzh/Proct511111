import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keplar.population.function import action_map


def find_turning_points(data, limit_num):
    turning_points = []

    for i in range(limit_num):
        tk = "data_gen" + str(i + 1)
        tk1 = "data_gen" + str(i + 2)
        if data[tk]["history_best_fit"] > data[tk1]["history_best_fit"]:
            turning_points.append(i + 1)
    return turning_points


def save(name='tmp', h=None):
    stype = "pdf"
    name = name.strip().replace(' ', '-').replace('%', 'pct')
    if h == None:
        h = plt.gcf()
    h.tight_layout()
    print('saving', name + f'.{stype}')
    plt.savefig(name + f'.{stype}', bbox_inches='tight')
    # plt.savefig(name + '.png')


# def print_actions_info(turning_points, data, json_data):
#     for i in turning_points:
#         plt.annotate(f'Actions: {json_data[f"data_gen{i + 1}"]["actions"]}',
#                      xy=(i, data[i]),
#                      xytext=(i, data[i] - 0.1),
#                      arrowprops=dict(facecolor='black', arrowstyle='->'))
#         actions_info = arr3[f"data_gen{i + 1}"]["actions"]
#         plt.text(i, rl_arr[i], f'Actions: {actions_info}', ha='center', va='bottom')
#         print(f'Actions at turning point generation{i + 1}:')
#         arr = json_data[f"data_gen{i + 1}"]["actions"]
#         for i in arr:
#             if i.isdigit():
#                 print({action_map[int(i)]})


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


# mat = readfile(open("/home/tzh/PycharmProjects/pythonProjectAR5/result/pmlb_1027_keplaroperon_generation.txt"))
# mat = np.array(mat)
# meancol = mat.mean(axis=0)
# print(meancol)
# arr2 = readfile(open("/home/tzh/PycharmProjects/pythonProjectAR5/result/pmlb_1027_keplarbingo_generation_fit.txt"))
# arr2 = np.array(arr2)
# meancol2 = arr2.mean(axis=0)
rd_arr1 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_random_test4.json")
rd_arr2 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_random_test5.json")
rd_arr3 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_random_test6.json")
rd_arr4 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_random_test7.json")
arr1 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_test29.json")
arr2 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_test31.json")
arr3 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_test32.json")
arr4 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_test33.json")
arr5 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_test34.json")
arr6 = pd.read_json("/home/tzh/PycharmProjects/pythonProjectAR5/result/RL6_test35.json")

fit_rl_arr1 = []
fit_rd_arr1 = []
fit_rl_arr2 = []
fit_rd_arr2 = []
fit_rl_arr3 = []
fit_rd_arr3 = []
fit_rl_arr4 = []
fit_rd_arr4 = []
fit_rl_arr5=  []
fit_rl_arr6= []
turning_points = find_turning_points(arr1, 169)
for i in range(170):
    tk = "data_gen" + str(i + 1)
    fit_rd_arr1.append(rd_arr1[tk]["history_best_fit"])
    fit_rl_arr1.append(arr1[tk]["history_best_fit"])
    fit_rl_arr2.append(arr2[tk]["history_best_fit"])
    fit_rd_arr2.append(rd_arr2[tk]["history_best_fit"])
    fit_rl_arr3.append(arr3[tk]["history_best_fit"])
    fit_rd_arr3.append(rd_arr3[tk]["history_best_fit"])
    fit_rl_arr4.append(arr4[tk]["history_best_fit"])
    fit_rd_arr4.append(rd_arr4[tk]["history_best_fit"])
    fit_rl_arr5.append(arr5[tk]["history_best_fit"])
    fit_rl_arr6.append(arr6[tk]["history_best_fit"])

# fit_rl_arr1 = np.array(fit_rl_arr1)
# fit_rd_arr1 = np.array(fit_rd_arr1)
# fit_rl_arr2 = np.array(fit_rl_arr2)
# fit_rd_arr2 = np.array(fit_rd_arr2)


x = np.arange(0, 170, 1)

# plt.plot(x, meancol2[0:130:1], label='KeplarBingo', color='red')
# plt.plot(x, meancol[0:250:1], label='KeplarOperon', color='blue')
plt.plot(x, fit_rl_arr1[0:170:1], label='Random_baseline', color='red')
plt.plot(x, fit_rd_arr1[0:170:1], label='Keplar_PPO_RL', color='green')
plt.plot(x, fit_rl_arr2[0:170:1], color='green')
plt.plot(x, fit_rd_arr2[0:170:1], color='red')
plt.plot(x, fit_rl_arr3[0:170:1], color='green')
plt.plot(x, fit_rd_arr3[0:170:1], color='red')
plt.plot(x, fit_rl_arr4[0:170:1], color='green')
plt.plot(x, fit_rd_arr4[0:170:1], color='red')
plt.plot(x, fit_rl_arr5[0:170:1], color='green')
plt.plot(x, fit_rl_arr6[0:170:1], color='green')
plt.legend(loc='best')
plt.xlabel('Generation')
plt.ylabel('TrainFitness(RMSE)')
plt.title(str(arr1["data_gen1"]["dataset"]) + ' x_dim=4 epoch=10')
# print_actions_info(turning_points, rl_arr, arr3)
if arr1["data_gen1"]["dataset"] != rd_arr1["data_gen1"]["dataset"]:
    raise ValueError("数据集不匹配")
fig_path = "/home/tzh/PycharmProjects/pythonProjectAR5/IMG_COLOR/LOG/" + str(rd_arr1["data_gen1"]["dataset"]) \
           + "rl_vs_rd_genfit.pdf"
plt.savefig(fig_path)
plt.show()
