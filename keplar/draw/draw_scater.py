import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import itertools


# 读取数据文件
def read_data(filename):
    data = np.genfromtxt(filename)
    return data


# 定义要拟合的曲面函数，这里使用二次多项式
def surface_func(coords, a, b, c, d, e, f):
    x, y = coords
    print(f"{a} * x ** 2 + {b} * y ** 2 + {d} * x + {c} * y + {e} * x * y + {f}")
    return a * x ** 2 + b * y ** 2 + d * x + c * y + e * x * y + f


# 绘制三维散点图和拟合曲面
def plot_3d_scatter_and_fit_surface(data, output_filename):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x, y, z, c='r', marker='o')

    # 拟合曲面
    popt, _ = curve_fit(surface_func, (x, y), z)
    a, b, c, d, e, f = popt
    xi, yi = np.meshgrid(x, y)
    zi = surface_func((xi, yi), a, b, c, d, e, f)

    # 绘制拟合的曲面
    ax.plot_surface(xi, yi, zi, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(str(output_filename))
    plt.savefig("IMG_COLOR/LOG/" + output_filename, format='pdf')
    plt.show()


def plot_3d_scatter(data, output_filename):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x, y, z, c='r', marker='o')

    # 拟合曲面

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title(str(output_filename))
    plt.savefig("IMG_COLOR/LOG/" + output_filename, format='pdf')

    plt.show()


# 生成所有列的排列组合
def generate_combinations(num_columns):
    return list(itertools.combinations(range(num_columns), 3))


if __name__ == "__main__":
    df = pd.read_csv("/home/tzh/PycharmProjects/pythonProjectAR5/datasets/pmlb/name.csv")
    print(df)
    df1 = df.iloc[:, -1]
    for j in df1:
        filename = "/home/tzh/PycharmProjects/pythonProjectAR5/datasets/pmlb/pmlb_txt/" + str(j) + ".txt"  # 替换为你的数据文件名
        data = read_data(filename)
        num_columns = data.shape[1]

        combinations = generate_combinations(num_columns)

        for i, combination in enumerate(combinations):
            output_filename = f"1_fit_surface_{i}.pdf"
            subset_data = data[:, combination]
            plot_3d_scatter_and_fit_surface(subset_data, output_filename)
