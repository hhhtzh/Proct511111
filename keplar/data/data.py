import numpy as np
import pandas as pd
from pmlb import fetch_data


class Data:
    def __init__(self, type, file_path):
        self.pd_data = None
        self.type = type
        self.file_path = file_path
        self.x = None
        self.y = None

    def read_file(self):
        if self.type == "csv":
            try:
                self.pd_data = pd.read_csv(self.file_path)
            except:
                raise ValueError('csv路径错误')
        elif self.type == "txt":
            self.pd_data = pd.read_table('path', sep='\t', header=None)
        elif self.type == "pmlb":
            try:
                self.pd_data = fetch_data(str(self.file_path), local_cache_dir='./datasets', return_X_y=False)
                self.x, self.y = fetch_data(str(self.file_path), local_cache_dir='./datasets', return_X_y=True)
            except:
                raise ValueError('pmlb数据集名称错误')
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

    def set_xy(self, str_y):
        self.y = np.array(self.pd_data.loc[str_y])
        self.x = np.array(self.pd_data.drop(lables=str_y, axis=1))

    def display_data(self):
        print(self.pd_data.head())

    def data_check(self):
        self.pd_data.info()

    def data_describe(self):
        print(self.pd_data.describe())
