import numpy as np

from keplar.operator.operator import Operator


class CheckPopulation(Operator):
    def __init__(self, data=None):
        super().__init__()
        self.data = data

    def do(self, population):
        length_list = []
        fit_list = []
        if population.pop_type != "self":
            raise ValueError("暂时不支持")
        else:
            for ind in population.pop_list:
                length_list.append(len(ind.func))
                fit_list.append(ind.fitness)
            np_length = np.array(length_list)
            np_fit = np.array(fit_list)
            max_length = np.max(np_length)
            min_length = np.min(np_length)
            mean_length = np.mean(np_length)
            best_fit = np.min(np_fit)
            worest_fit = np.max(np_fit)
            mean_fit = np.mean(np_fit)
            return [best_fit, worest_fit, mean_fit, max_length, min_length, mean_length]
