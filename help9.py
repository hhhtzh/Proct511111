import csv

import pandas as pd

from keplar.data.data import Data
from keplar.operator.evaluator import SingleBingoEvaluator, OperonSingleEvaluator

# data = Data("txt", "datasets/pmlb/val/197_cpu_act.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","y"])
# data = Data("txt_pmlb", "datasets/pmlb/val/529_pollen.txt",
#             ["x0", "x1", "x2", "x3", "y"])
# data = Data("txt_pmlb", "datasets/feynman/train/feynman-bonus.8.txt", ["x0", "x1", "y"])
# data = Data("txt_pmlb", "datasets/pmlb/train/588_fri_c4_1000_100.txt", ["x0", "x1", "x2", "x3", "y"])
data = Data("txt_pmlb", "/home/tzh/PycharmProjects/pythonProjectAR5/datasets/pmlb/val/620_fri_c1_1000_25.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24","y"])
data.read_file()
data.set_xy("y")
# npx=data.get_np_x()
# print(npx)
fit_list = []
with open('result/pmlb_294_satellite_image_equ.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
i = 1
x=data.get_np_x()
y=data.get_np_y()
while i < len(rows):
    print(rows[i][1])
    eval = OperonSingleEvaluator("R2", x, y, 1, True, op_equ=rows[i][1])
    fit = eval.do()
    fit_list.append(fit)
    i += 1
fit_pd = pd.DataFrame({'MTaylor(with new sparse)': fit_list})
fit_pd.to_csv(r"result/pmlb_620_fri_c1_1000_25_fit.csv", sep=',', mode="a")
