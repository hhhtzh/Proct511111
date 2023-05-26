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

    def get_x(self, int_x):
        d = self.pd_data.columns[int_x:int_x + 1]
        return np.array(self.pd_data[d])

    def get_y(self, int_y):
        d = self.pd_data.columns[int_y:int_y + 1]
        return np.array(self.pd_data[d])

    def get_pd_x(self, int_x):
        d = self.pd_data.columns[int_x:int_x + 1]
        return self.pd_data[d]

    def get_pd_y(self, int_y):
        d = self.pd_data.columns[int_y:int_y + 1]
        return self.pd_data[d]

    def display_data(self):
        self.pd_data.head()

    def data_check(self):
        self.pd_data.info()

    def data_describe(self):
        self.pd_data.describe()
