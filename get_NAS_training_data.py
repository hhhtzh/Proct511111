import csv

from keplar.Algorithm.Alg import KeplarBingoAlg, GpBingo2Alg, OperonBingoAlg
from keplar.data.data import Data
from keplar.operator.check_pop import CheckPopulation
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import GpCreator, OperonCreator, BingoCreator
from keplar.operator.crossover import BingoCrossover, OperonCrossover
from keplar.operator.evaluator import BingoEvaluator, OperonEvaluator, GpEvaluator
from keplar.operator.mutation import BingoMutation, OperonMutation
from keplar.operator.selector import BingoSelector

# KeplarBingo=1,GpBingo=2,OperonBingo=3
data = Data("pmlb", "1027_ESL", ["x0", "x1", "x2", "x3", 'y'])
# data = Data("pmlb", "529_pollen", ["x1", "x2", "x3", "x4", 'y'])
# data = Data("pmlb", "analcatdata_creditscore", ["x0", "x1", "x2", "x3", "x4", "x5", "y"])
# data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
#
data.read_file()
# data.set_xy("y")
x = data.get_np_x()
y = data.get_np_y()
op_creator = OperonCreator("balanced", x, y, 128, "Operon")
evaluator = OperonEvaluator("RMSE", x, y, 0.7, True, "self")
population = op_creator.do()
op_crossover = OperonCrossover(x, y, "Operon")
op_mutation = OperonMutation(0.6, 0.7, 0.8, 0.8, x, y, 10, 50, "balanced", "Operon")
list2 = []
operators = ["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
bg_creator = BingoCreator(128, operators, x, 10, "Bingo")
bg_evaluator = BingoEvaluator(x, "exp", "lm", "self", y)
bg_crossover = BingoCrossover("Bingo")
bg_mutation = BingoMutation(x, operators, "Bingo")
bg_selector = BingoSelector(0.5, "tournament", "Bingo")
opbg_select = BingoSelector(0.2, "tournament", "Operon")
kb_gen_up_oplist = CompositeOp([bg_crossover, bg_mutation])
kb_gen_down_oplist = CompositeOpReturn([bg_selector])
kb_gen_eva_oplist = CompositeOp([bg_evaluator])
gp_evaluator = GpEvaluator(x, y, "self", metric="rmse")
gen_up_oplist = CompositeOp([bg_crossover, bg_mutation])
gen_down_oplist = CompositeOpReturn([bg_selector])
gen_eva_oplist = CompositeOp([gp_evaluator])
op_up_list = [op_mutation, op_crossover]
eval_op_list = [evaluator]
evaluator = OperonEvaluator("RMSE", x, y, 0.7, True, "self")
eval_op_list = [evaluator]
select = BingoSelector(0.2, "tournament", "Operon")
crossover = OperonCrossover(x, y, "Operon")
mutation = OperonMutation(0.6, 0.7, 0.8, 0.8, x, y, 10, 50, "balanced", "Operon")
op_up_list = [mutation, crossover]

# alg = OperonBingoAlg(1, op_up_list, None, eval_op_list, -10, population, select, x, y, 128)
# alg.one_gen_run()

for _ in range(1000):
    temp_fit_list = []
    temp_pop_list = []
    best_index = 0
    # print(population.pop_type)
    evaluator.do(population)
    # for ind in population.pop_list:
    #     print(ind.fitness)
    ck = CheckPopulation(data)
    list1 = ck.do(population)
    population1 = population
    population2 = population
    population3 = population
    bgsr = KeplarBingoAlg(1, kb_gen_up_oplist, kb_gen_down_oplist, kb_gen_eva_oplist, 0.001, population1)
    bgsr.one_gen_run()
    gpbg2 = GpBingo2Alg(1, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population2)
    gpbg2.one_gen_run()
    opbg = OperonBingoAlg(1, op_up_list, None, eval_op_list, -10, population3, select, x, y, 128)
    opbg.one_gen_run()
    temp_pop_list.append(population1)
    temp_pop_list.append(population2)
    temp_pop_list.append(population3)
    temp_fit_list.append(bgsr.best_fit)
    temp_fit_list.append(gpbg2.best_fit)
    temp_fit_list.append(opbg.best_fit)
    best_fit = 100
    # print("开始比较")
    # print(temp_fit_list)
    for i in range(len(temp_fit_list)):
        if temp_fit_list[i] < best_fit:
            best_fit = temp_fit_list[i]
            best_index = i
    list1.append(best_index + 1)
    list2.append(list1)
    population = temp_pop_list[best_index]

header = ["BestFit", 'WorstFit', 'MeanFit', 'Longest', 'shortest', 'MeanLength', 'SelectedLable']
with open('NAStraining_data/recursion_training2.csv', 'w', encoding='utf-8') as file_obj:
    writer = csv.writer(file_obj)
    writer.writerow(header)
    for i in list2:
        writer.writerow(i)
