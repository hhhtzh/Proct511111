# from pmlb import fetch_data

# from keplar.operator.cleaner import BingoCleaner
# from keplar.operator.composite_operator import CompositeOp
# from keplar.operator.creator import BingoCreator, GpCreator
# from keplar.operator.crossover import BingoCrossover
# from keplar.operator.evaluator import BingoEvaluator
# from keplar.operator.generator import BingoGenerator
# from keplar.operator.mutation import BingoMutation
# from keplar.operator.selector import BingoSelector

from keplar.Algorithm.dsr_Algorithm import uDsrAlgorithm


if __name__ == '__main__':
    csv_filename = "/Users/friday/Documents/lab_work/SR_for_Bozhou/Eight/pythonProjectAR5/datasets/data.csv"
    config_filename = "/Users/friday/Documents/lab_work/SR_for_Bozhou/Eight/pythonProjectAR5/datasets/config_regression.json"
    udsr = uDsrAlgorithm(csv_filename,config_filename)
    udsr.run()