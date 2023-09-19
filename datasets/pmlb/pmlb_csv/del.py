import csv
import shutil

# 输入文件的文件名
filename = "1203_BNG_pwLinear.csv"

# 打开原始文件并读取内容
with open(filename, "r", newline="") as file:
    reader = csv.reader(file)
    header = next(reader)
    data = list(reader)

# 计算要保留的行数
total_rows = len(data)
rows_to_keep = total_rows // 4  # 保留前四分之一的行数

# 重新写入原始文件
with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # 写入标题行
    writer.writerows(data[:rows_to_keep])  # 写入前四分之一的行

print("已删除CSV文件的后五分之四并保存到原文件。")

