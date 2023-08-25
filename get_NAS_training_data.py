import csv

from keplar.Algorithm.Alg import KeplarBingoAlg, GpBingo2Alg, OperonBingoAlg
from keplar.data.data import Data
from keplar.operator.check_pop import CheckPopulation
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import GpCreator, OperonCreator, BingoCreator
from keplar.operator.crossover import BingoCrossover
from keplar.operator.evaluator import BingoEvaluator, OperonEvaluator, GpEvaluator
from keplar.operator.mutation import BingoMutation
from keplar.operator.selector import BingoSelector

# KeplarBingo=1,GpBingo=2,OperonBingo=3
data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
x = data.get_np_x()
y = data.get_np_y()
op_creator = OperonCreator("balanced", x, y, 128, "Operon")
evaluator = OperonEvaluator("RMSE", x, y, 0.7, True, "self")
population = op_creator.do()

list2 = []
operators = ["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
bg_creator = BingoCreator(128, operators, x, 10, "Bingo")
bg_evaluator = BingoEvaluator(x, "exp", "lm", "self", y)
bg_crossover = BingoCrossover("Bingo")
bg_mutation = BingoMutation(x, operators, "Bingo")
bg_selector = BingoSelector(0.5, "tournament", "Bingo")
kb_gen_up_oplist = CompositeOp([bg_crossover, bg_mutation])
kb_gen_down_oplist = CompositeOpReturn([bg_selector])
kb_gen_eva_oplist = CompositeOp([bg_evaluator])
gp_evaluator = GpEvaluator(x, y, "Bingo", metric="rmse")
gen_up_oplist = CompositeOp([bg_crossover, bg_mutation])
gen_down_oplist = CompositeOpReturn([bg_selector])
gen_eva_oplist = CompositeOp([gp_evaluator])
for _ in range(1000):
    evaluator.do(population)
    ck = CheckPopulation(data)
    list1 = ck.do(population)
    print(list1)
    population1=population
    population2=population
    population3=population
    bgsr = KeplarBingoAlg(1, kb_gen_up_oplist, kb_gen_down_oplist, kb_gen_eva_oplist, 0.001, population1)
    bgsr.one_gen_run()
    gpbg2=GpBingo2Alg(1, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population2)
    gpbg2.one_gen_run()
    opbg=OperonBingoAlg(1, op_up_list, None, eval_op_list, -10, population, select, x, y, 128)


header = ["BestFit", 'WorstFit', 'MeanFit', 'Longest', 'shortest', 'MeanLength', 'SelectedLable']
with open('NAStraining_data/recursion_training.csv', 'w', encoding='utf-8') as file_obj:
    writer = csv.writer(file_obj)
    writer.writerow(header)
    for i in list2:
        writer.writerow(i)
