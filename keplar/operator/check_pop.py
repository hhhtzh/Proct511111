import numpy as np

from keplar.operator.operator import Operator


class CheckPopulation(Operator):
    def __init__(self,data=None):
        super().__init__()
        self.data = data

    def do(self,population):
        length_list=[]
        if population.pop_type!="self":
            raise ValueError("暂时不支持")
        else:
            for ind in population.pop_list:
                length_list.append(len(ind.func))
            np_length=np.array(length_list)
            max_length=np_length.argmax()
            min_length=np_length.argmin()
            mean_length=np_length.mean()
            print(max_length)
            print(min_length)
            print(mean_length)



