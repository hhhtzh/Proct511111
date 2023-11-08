import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置Seaborn风格
sns.set_theme(style="darkgrid")

# 创建一个示例数据集，或者加载你自己的数据
# 这里使用Seaborn内置的 "fmri" 数据集作为示例
# 如果你有自己的数据，可以用类似的方式加载
fmri = sns.load_dataset("fmri")
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
attributes = pd.Series(['random_baseline'] * 4 + ['Keplar_PPO_RL'] * 5, name='Attribute')
rd_arr1 = rd_arr1.transpose()
rd_arr2 = rd_arr2.transpose()
rd_arr3 = rd_arr3.transpose()
rd_arr4 = rd_arr4.transpose()
arr1 = arr1.transpose()
arr2 = arr2.transpose()
arr3 = arr3.transpose()
arr4 = arr4.transpose()
arr5 = arr5.transpose()
arr6 = arr6.transpose()
print(arr6)
print(attributes)
rd_arr1['Attribute'] = "random_baseline"
rd_arr2['Attribute'] = "random_baseline"
rd_arr3['Attribute'] = "random_baseline"
rd_arr4['Attribute'] = "random_baseline"
arr1['Attribute'] = 'Keplar_PPO_RL'
arr2['Attribute'] = 'Keplar_PPO_RL'
arr3['Attribute'] = 'Keplar_PPO_RL'
arr4['Attribute'] = 'Keplar_PPO_RL'
arr5['Attribute'] = 'Keplar_PPO_RL'
arr6['Attribute'] = 'Keplar_PPO_RL'
merged_df = pd.concat([rd_arr1, rd_arr2, rd_arr3, rd_arr4, arr1, arr2, arr3, arr4, arr5, arr6], ignore_index=True)
print(merged_df)


# 绘制线图
plt.figure(figsize=(10, 6))  # 设置图形大小

# 使用Seaborn的lineplot函数绘制线图
# sns.lineplot(x="timepoint", y="signal",  # x和y轴的数据列
#              hue="region", style="event",  # 按不同的列进行着色和样式区分
#              data=fmri)
sns.lineplot(x="generation", y="history_best_fit",  # x和y轴的数据列
             hue="Attribute", style="Attribute",  # 按不同的列进行着色和样式区分
             data=merged_df)

# 添加标题和标签
plt.title("pmlb_1027 x_dim=4 epoch=10")
plt.xlabel("generation")
plt.ylabel("history_best_fit")

# 显示图例
plt.legend(title="Legend", loc="upper right")
fig_path = "/home/tzh/PycharmProjects/pythonProjectAR5/IMG_COLOR/LOG/" + "pmlb_1027" \
           + "rl_vs_rd_genfit_seaborn.pdf"
plt.savefig(fig_path)

# 显示图表
plt.show()
