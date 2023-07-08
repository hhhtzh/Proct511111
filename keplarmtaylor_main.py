from keplar.Algorithm.operon_Algorithm import OperonAlg
from keplar.data.data import Data
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.reinserter import OperonReinserter
from keplar.operator.selector import OperonSelector
from keplar.operator.taylor_judge import TaylorJudge
from keplar.preoperator.ml.sklearndbscan import SklearnDBscan

data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")

for i in [1e-5, 0.2, 1, 4, 10, 100]:
    dbscan = SklearnDBscan(eps=i)
    x = dbscan.do(data)
    if x:
        break
db_sum = x
programs = []
fit_list = []
for db_i in db_sum:
    taylor = TaylorJudge(db_i, "taylorgp")
    jd = taylor.do()
    if jd == "end":
        programs.append(taylor.program)
        fit_list.append(taylor.end_fitness)
    else:
        selector = OperonSelector(5)
        evaluator = OperonEvaluator("RMSE", x, y, 0.5, True, "Operon")
        crossover = OperonCrossover(x, y, "Operon")
        mutation = OperonMutation(1, 1, 1, 0.5, x, y, 10, 50, "balanced", "Operon")
        reinsert = OperonReinserter(None, "ReplaceWorst", 10, "Operon", x, y)
        op_up_list = [mutation, crossover]
        op_down_list = [reinsert]
        eva_list = [evaluator]
        op_alg = OperonAlg(1000, op_up_list, op_down_list, eva_list, selector, 1e-5, 1000, 16, x, y)
        op_alg.run()
