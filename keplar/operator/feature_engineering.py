import time

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from TaylorGP.src.taylorGP.calTaylor import Metrics
from keplar.operator.operator import Operator


class FeatureEngineering(Operator):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def do(self, population=None):
        raise NotImplemented


class SklearnFeature(FeatureEngineering):
    def __init__(self, method, data):
        super().__init__(data)
        self.method = method

    def do(self, population=None):
        if "PCA" in self.method:
            p = PCA()
            pd = self.data.pd_data
            self.data.pd_data = p.fit_transform(pd)
        if "filter" in self.method:
            var = VarianceThreshold(threshold=1)
            pd = self.data.pd_data
            self.data.pd_data = var.fit_transform(pd)


class TaylorFeature(FeatureEngineering):
    def __init__(self, data, output_fileName):
        super().__init__(data)
        self.output_fileName = output_fileName

    def do(self, population=None):
        eqName = r'eq_' + str(self.output_fileName) + '.csv'
        # eqName = r'result/eq_' + str(fileNum) + '.csv'
        eq_write = open(eqName, "w+")  # 重新写
        eq_write.write(
            'Gen|Top1|Length1|Fitness1|Top2|Length2|Fitness2|Top3|Length3|Fitness3|Top4|Length4|Fitness4|Top5|Length5|Fitness5|Top6|Length6|Fitness6|Top7|Length7|Fitness7|Top8|Length8|Fitness8|Top9|Length9|Fitness9|Top10|Length10|Fitness10\n')
        print('eqName=', self.output_fileName)
        average_fitness = 0
        repeat = 1
        time_start1 = time.time()
        for repeatNum in range(repeat):
            time_start2 = time.time()
            X = self.data.get_np_x()
            Y = self.data.get_np_y()
            np_ds = self.data.get_np_ds()
            loopNum = 0
            Metric = []
            while True:  # 进行5次低阶多项式的判别--不是则继续
                metric = Metrics(varNum=X.shape[1], dataSet=np_ds)
                loopNum += 1
                Metric.append(metric)
                if loopNum == 2 and X.shape[1] <= 2:
                    break
                elif loopNum == 5 and (2 < X.shape[1] <= 3):
                    break
                elif loopNum == 4 and (3 < X.shape[1] <= 4):
                    break
                elif loopNum == 3 and (4 < X.shape[1] <= 5):
                    break
                elif loopNum == 2 and (5 < X.shape[1] <= 6):
                    break
                elif loopNum == 1 and (X.shape[1] > 6):
                    break
            Metric.sort(key=lambda x: x.nmse)  # 按低阶多项式拟合误差对20个metric排序
            metric = Metric[0]
            print('排序后选中多项式_and_低阶多项式MSE:', metric.nmse, metric.low_nmse)
            eq_write.write(
                str(-1) + '|' + str(metric.f_low_taylor) + '|' + '10' + '|' + str(metric.low_nmse) + '|' + '\n')
            if metric.nmse < 0.1:
                metric.nihe_flag = True
            else:  # 拟合失败后Taylor特征的判别以数据集为准
                metric.bias = 0.
                print('拟合失败')
            Pop = 1000
            end_fitness, program = None, None



