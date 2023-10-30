# import pandas as pd
#
# ft = pd.read_feather("pmlb_results.feather")
# ft1 = pd.read_feather("feynman_results.feather")
#
# # ... 其他操作 ...
#
# # 创建新的 DataFrame
# new_l = pd.DataFrame([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#
# # 将新数据与现有数据合并
# ft1 = pd.concat([ft1, new_l], ignore_index=True)
#
# # 将合并后的 DataFrame 写入 feather 文件
# ft1.to_feather("feynman_results.feather")
# print(ft1)

# import pandas as pd
# ft1 = pd.read_feather("feynman_results.feather")
# # 创建要追加的新数据行
# new_row_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# new_row = pd.DataFrame([new_row_data], columns=ft1.columns)
#
# # 读取现有的 feather 文件
#
#
# # 将新数据行追加到 DataFrame 中
# ft1 = ft1.append(new_row, ignore_index=True)
#
# # 将更新后的 DataFrame 重新写入 feather 文件
# ft1.to_feather("feynman_results.feather")


import pandas as pd

import sys
import argparse
import os


from bingo.symbolic_regression.agraph.string_parsing import eq_string_to_infix_tokens


sys.setrecursionlimit(10000)
argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="620_fri_c1_1000_25")

args = argparser.parse_args()
fileName_whitout_ext = args.dataset
# fileName = os.path.basename(args.dataset)
# string1 = os.path.splitext(fileName)[0]
# # string = "620_fri_c1_1000_25_time"
# fileName_whitout_ext = string1.replace("_time", "")


# print("file path : ",args.dataset)
# print("file name : ",fileName)
# print("file name without ext : ",fileName_whitout_ext)

file_type = "pmlb"
# if "feynman" in fileName_whitout_ext:
#     file_type = "feynman"
# else:
#     file_type = "pmlb"
#
# # 读取现有的 feather 文件
if file_type == "feynman":
    ft = pd.read_feather("feynman_results.feather")
else:
    ft = pd.read_feather("pmlb_results.feather")
#
#
# time = pd.read_csv("../../zjw_result/"+fileName_whitout_ext+"_time.csv")
# equ = pd.read_csv("../../zjw_result/"+fileName_whitout_ext+"_equ.csv")
# fit = pd.read_csv("../../zjw_result/"+fileName_whitout_ext+"_R2.csv")

# print(equ)
# time1 = time.iloc[:, -1]
# equ1 = equ.iloc[:, -1]
# fit1 = fit.iloc[:, -1]
# for i in equ1:
#     list1 = eq_string_to_infix_tokens(str(equ1))
#     print(list1)
#
# for i in range(len(time1)):
#     ms = 0
#     for j in equ1[i]:
#         if j in ['+', '-', '*', '/', '**']:
#             ms += 1
#     print(float(fit1[i]))
#     new_row_data = {
#         "dataset": fileName_whitout_ext,
#         "algorithm": "mtaylor",
#         "model_size": ms,
#         "training time (s)": float(time1[i]),
#         "r2_test": float(fit1[i]),
#     }
#     new_row = pd.DataFrame([new_row_data])
#     ft = pd.concat([ft, new_row], ignore_index=True)
#     if file_type == "feynman":
#         ft.to_feather("feynman_results.feather")
#     else:
#         ft.to_feather("pmlb_results.feather")

# ft1 = ft1.drop(ft1.index[-1])
#
# # 将更新后的 DataFrame 重新写入 feather 文件
# ft1.to_feather("feynman_results.feather")
print(ft)
print(ft.iloc[-1])
# print(ft1.loc[ft1["dataset"] == "225_puma8NH"]["r2_zero_test"])
# print(ft1.loc[ft1["dataset"] == "225_puma8NH"]["algorithm"])
# print(ft.loc[ft["dataset"] == "feynman_II_15_5"]["mtaylor"])
# print(ft.loc[ft["dataset"] == "620_fri_c1_1000_25"])
# # 创建要追加的新数据行

# new_row = pd.DataFrame([new_row_data])
#
# # 将新数据行与现有数据合并
# ft1 = pd.concat([ft1, new_row], ignore_index=True)
#
# # 将更新后的 DataFrame 重新写入 feather 文件
# ft1.to_feather("feynman_results.feather")
