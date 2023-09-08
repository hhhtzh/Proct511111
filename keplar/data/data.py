import numpy as np
import pandas as pd
from pmlb import fetch_data


class Data:
    def __init__(self, type, file_path, names):
        self.names = names
        self.pd_data = None
        self.np_data = None
        self.type = type
        self.file_path = file_path
        self.x = None
        self.y = None

    def read_file(self):
        np.set_printoptions(suppress=True)
        if self.type == "csv":
            try:
                self.pd_data = pd.read_csv(self.file_path)
            except:
                raise ValueError('csv路径错误')
        elif self.type == "txt":
            self.pd_data = pd.DataFrame(
                pd.read_table(self.file_path, sep='  ', header=None, engine='python', names=self.names))
        elif self.type == "txt_pmlb":
            self.pd_data = pd.DataFrame(
                pd.read_table(self.file_path, sep=' ', header=None, engine='python', names=self.names))
        elif self.type == "pmlb":
            try:
                self.pd_data = fetch_data(str(self.file_path), local_cache_dir='./datasets', return_X_y=False)
                self.x, self.y = fetch_data(str(self.file_path), local_cache_dir='./datasets', return_X_y=True)
            except:
                raise ValueError('pmlb数据集名称错误')
        elif self.type == "numpy":
            try:
                self.pd_data = pd.DataFrame(self.file_path)
                self.x, self.y = pd.DataFrame(self.file_path[:, :-1]), pd.DataFrame(self.file_path[:, -1])
            except:
                raise ValueError('numpy数据集设置失败')
        elif self.type == "tsv":
            try:
                self.pd_data = pd.read_csv(str(self.file_path), sep='\t')
                self.x = self.pd_data.iloc[:, :-1]  # 提取除最后一列外的所有列作为特征
                self.y = self.pd_data.iloc[:, -1]  # 提取最后一列作为目标变量
            except:
                raise ValueError('TSV文件读取失败')
        else:
            raise ValueError("获取数据的方法选择错误")

    def get_feature(self):
        self.ft_str = list()
        for dt1 in self.pd_data.columns:
            self.ft_str.append(str(dt1))

    def get_col(self, str):
        return self.pd_data.loc[str]

    def get_x(self):
        if self.x is not None:
            return self.x
        else:
            raise ValueError("数据集xy未设置")

    def get_y(self):
        if self.y is not None:
            return self.y
        else:
            raise ValueError("数据集xy未设置")

    def get_np_x(self):
        if self.x is not None:
            np.set_printoptions(suppress=True)
            return np.array(self.x)
        else:
            raise ValueError("数据集xy未设置")

    def get_np_y(self):
        if self.y is not None:
            np.set_printoptions(suppress=True)
            return np.array(self.y)
        else:
            raise ValueError("数据集xy未设置")

    def set_xy(self, str_y):
        self.y = np.array(self.pd_data.loc[:, str_y])
        dt = pd.DataFrame(self.pd_data)
        self.x = np.array(dt.drop(labels=str_y, axis=1))

    def get_np_ds(self):
        np_x = self.get_np_x()
        np_y = self.get_np_y()
        np_y = np_y.reshape([-1, 1])
        self.np_data = np.hstack([np_x, np_y])
        return self.np_data

    # 默认最后一列为y

    def display_data(self):
        print(self.pd_data.head())

    def data_check(self):
        self.pd_data.info()

    def data_describe(self):
        print(self.pd_data.describe())
