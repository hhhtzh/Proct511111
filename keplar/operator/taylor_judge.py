import time

from sympy import symbols

from TaylorGP.TaylorGP import CalTaylorFeatures
from TaylorGP.src.taylorGP.calTaylor import Metrics
from keplar.operator.operator import Operator


class TaylorJudge(Operator):
    def __init__(self, data, output_fileName):
        super().__init__()
        self.data = data
        self.program = None
        self.end_fitness = None
        self.metric = None
        self.output_fileName = output_fileName

    def do(self,population=None):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29 = symbols(
            "x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,"
            "x28,x29 ")
        _x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22,
              x23, x24, x25, x26, x27, x28, x29]
        eqName = r'eq_' + str(self.output_fileName) + '.csv'
        # eqName = r'result/eq_' + str(fileNum) + '.csv'
        eq_write = open(eqName, "w+")  # 重新写
        eq_write.write(
            'Gen|Top1|Length1|Fitness1|Top2|Length2|Fitness2|Top3|Length3|Fitness3|Top4|Length4|Fitness4|Top5|Length5'
            '|Fitness5|Top6|Length6|Fitness6|Top7|Length7|Fitness7|Top8|Length8|Fitness8|Top9|Length9|Fitness9|Top10'
            '|Length10|Fitness10\n')
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
            self.metric = Metric[0]
            print('排序后选中多项式_and_低阶多项式MSE:', self.metric.nmse, self.metric.low_nmse)
            eq_write.write(
                str(-1) + '|' + str(self.metric.f_low_taylor) + '|' + '10' + '|' + str(
                    self.metric.low_nmse) + '|' + '\n')
            if self.metric.nmse < 0.1:
                self.metric.nihe_flag = True
            else:  # 拟合失败后Taylor特征的判别以数据集为准
                self.metric.bias = 0.
                print('拟合失败')
            Pop = 1000
            self.end_fitness, self.program = None, None
            # 下面分成有可分性的递归和非递归处理方式---对应需要Metric和Metric2两个类分别处理,Metric中的_x都是默认的
            if metric.judge_Low_polynomial():
                self.end_fitness, self.program = metric.low_nmse, metric.f_low_taylor
                return "end"
            elif metric.nihe_flag and (metric.judge_additi_separability() or metric.judge_multi_separability()):
                self.end_fitness, self.program = CalTaylorFeatures(metric.f_taylor, _x[:X.shape[1]], X, Y, Pop,
                                                                   repeatNum,
                                                                   eq_write)
                return "end"
                # nihe_flag, low_polynomial, qualified_list, Y = cal_Taylor_features(varNum=X.shape[1], fileNum=fileNum, Y=Y)
            else:  # 针对多变量不能拟合：仅根据数据集获取边界以及单调性指导SR--放在Taylor特征判别处理

                self.end_fitness, self.program = metric.low_nmse, metric.f_low_taylor
                return "not end"