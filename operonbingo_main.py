from keplar.data.data import Data
from keplar.operator.creator import OperonCreator
from keplar.operator.crossover import OperonCrossover
from keplar.operator.evaluator import OperonEvaluator
from keplar.operator.mutation import OperonMutation
from keplar.operator.reinserter import OperonReinserter
from keplar.operator.selector import BingoSelector

data = Data("pmlb", '1027_ESL')
data.read_file()
x = data.get_x()
y = data.get_y()
creator = OperonCreator("balanced", x, y, 128, "Operon")
population = creator.do()
evaluator = OperonEvaluator("R2", x, y, 0.5, True)
eva_op_list = [evaluator]
select = BingoSelector(0.4, "tournament", "Operon")
crossover = OperonCrossover()
mutation = OperonMutation()
reinsert = OperonReinserter()
op_up_list = [mutation, crossover]
op_down_list = [reinsert]
