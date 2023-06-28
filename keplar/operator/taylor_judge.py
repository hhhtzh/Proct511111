from TaylorGP.TaylorGP import CalTaylorFeatures
from keplar.operator.operator import Operator


class TaylorJudge(Operator):
    def __init__(self,taylor_metric):
        super().__init__()
        self.taylor_metric = taylor_metric

    def do(self,population=None):
        # 下面分成有可分性的递归和非递归处理方式---对应需要Metric和Metric2两个类分别处理,Metric中的_x都是默认的
        if self.taylor_metric.judge_Low_polynomial():
            end_fitness, program = self.taylor_metric.low_mse, self.taylor_metric.f_low_taylor
        elif self.taylor_metric.nihe_flag and (self.taylor_metric.judge_additi_separability() or self.taylor_metric.judge_multi_separability()):
            end_fitness, program = CalTaylorFeatures(self.taylor_metric.f_taylor, _x[:X.shape[1]], X, Y, Pop, repeatNum, eq_write)
            # nihe_flag, low_polynomial, qualified_list, Y = cal_Taylor_features(varNum=X.shape[1], fileNum=fileNum, Y=Y)
        else:  # 针对多变量不能拟合：仅根据数据集获取边界以及单调性指导SR--放在Taylor特征判别处理

            qualified_list = []
            qualified_list.extend(
                [self.taylor_metric.judge_Bound(),
                 self.taylor_metric.f_low_taylor,
                 self.taylor_metric.low_mse,
                 self.taylor_metric.bias,
                 self.taylor_metric.judge_parity(),
                 self.taylor_metric.judge_monotonicity()])
            print(qualified_list)
            end_fitness, program = Taylor_Based_SR(_x, X, metric.change_Y(Y), qualified_list, eq_write, Pop, repeatNum,
                                                   metric.low_mse < 1e-5)
