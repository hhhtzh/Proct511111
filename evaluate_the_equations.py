import csv

import pandas as pd

from keplar.data.data import Data
from keplar.operator.evaluator import SingleBingoEvaluator, OperonSingleEvaluator

# data = Data("txt", "datasets/pmlb/val/197_cpu_act.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","y"])
# data = Data("txt_pmlb", "datasets/pmlb/val/529_pollen.txt",
#             ["x0", "x1", "x2", "x3", "y"])
# data = Data("txt_pmlb", "datasets/feynman/val/feynman-bonus.8.txt", ["x0", "x1", "y"])
# data = Data("txt_pmlb", "datasets/pmlb/val/529_pollen.txt", ["x0", "x1", "x2", "x3", "y"])
data = Data("txt_pmlb", "datasets/feynman/val/feynman-i.12.1.txt", ["x0", "x1", "y"])
# data = Data("txt_pmlb", "datasets/pmlb/val/294_satellite_image.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29","x30", "x31", "x32", "x33", "x34", "x35", "y"])
# data = Data("txt_pmlb", "datasets/pmlb/val/607_fri_c4_1000_50.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29","x30", "x31", "x32", "x33", "x34", "x35","x36", "x37", "x38", "x39", "x40","x41", "x42", "x43", "x44", "x45", "x46", "x47", "x48", "x49", "y"])
# data = Data("txt_pmlb", "datasets/pmlb/val/608_fri_c3_1000_10.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y"])
# data = Data("txt_pmlb", "datasets/pmlb/val/612_fri_c1_1000_5.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
# data = Data("txt_pmlb", "datasets/feynman/val/feynman-i.12.1.txt", ["x0", "x1", "y"])
data.read_file()
data.set_xy("y")
# npx=data.get_np_x()
# print(npx)
fit_list = []
with open('/home/tzh/PycharmProjects/pythonProjectAR5/result/feynman-i.12.1_equ.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
i = 1
x = data.get_np_x()
y = data.get_np_y()
while i < len(rows):
    print(rows[i][1])
    eval = OperonSingleEvaluator("R2", x, y, 1, True, op_equ=rows[i][1])
    # eval = SingleBingoEvaluator(data, equation=rows[i][1], metric="mae")
    fit = eval.do()
    fit_list.append(fit)
    i += 1
fit_pd = pd.DataFrame({'MTaylor': fit_list})
fit_pd.to_csv(r"/home/tzh/PycharmProjects/pythonProjectAR5/result/feynman-i.12.1_fit.csv", sep=',', mode="a")
