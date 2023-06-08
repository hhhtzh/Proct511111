from keplar.operator.operator import Operator
import pyoperon as Operon


class Reinserter(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        raise NotImplementedError

class OperonReinserter(Reinserter):
    def __init__(self,pool,method,comparision_size):
        super().__init__()
        self.comparision_size = comparision_size
        self.method = method
        self.pool = pool


    def do(self,population):
        if self.comparision_size>population.pop_size:
            raise ValueError("比较数量大于种群数量")
        if not isinstance(self.comparision_size,int):
            raise ValueError("比较数量必须为int类型")
        if self.method=="ReplaceWorst":
            rein=Operon.ReplaceWorstReinserter(self.comparision_size)
        elif self.method=="KeepBest":
            rein=Operon.KeepBestReinserter(self.comparision_size)
        else:
            raise ValueError("reinserter方法选择错误")
        if population.pop_type=="Operon":
            if len(population.target_pop_list)!=len(population.target_fit_list):
                raise ValueError("个体与适应度数量不符")
        else:
            pass




