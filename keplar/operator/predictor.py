import numpy as np

from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression import ExplicitTrainingData, ExplicitRegression, AGraph
from keplar.operator.evaluator import Evaluator


class MetricsBingoPredictor(Evaluator):
    def __init__(self, data, func_fund_list, metric="rmse", optimizer_method="lm"):
        super().__init__()
        self.optimizer_method = optimizer_method
        self.metric = metric
        self.func_fund_list = func_fund_list
        self.data = data

    def do(self, population=None):
        global xtrain
        x = self.data.get_np_x()
        y = self.data.get_np_y()
        training_data = ExplicitTrainingData(x, y)
        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
        if self.optimizer_method not in ["lm", "TNC", "BFGS", "L-BFGS-B", "CG", "SLSQP"]:
            raise ValueError("优化方法名称未识别")
        optimizer = ScipyOptimizer(fitness, method=self.optimizer_method)
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)
        print("len_func_fund_list"+str(len(self.func_fund_list)))
        for i in range(len(self.func_fund_list)):
            for a, j in enumerate(self.func_fund_list[i]):
                bingo_ind = AGraph(equation=str(j))
                bingo_ind._update()
                arr = np.array(bingo_ind.evaluate_equation_at(x)).reshape(-1, 1)
                print(np.shape(arr))
            if i == 0:
                xtrain = arr
            else:
                xtrain=np.append(xtrain, arr, axis=1)
        return xtrain
