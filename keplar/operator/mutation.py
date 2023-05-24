from bingo.symbolic_regression import ComponentGenerator, AGraph
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from keplar.population.individual import Individual
from keplar.operator.operator import Operator
import numpy as np


class Mutation(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        pass


class BingoMutation(Mutation):
    def __init__(self, x, operators, command_probability=0.2, node_probability=0.2, parameter_probability=0.2,
                 prune_probability=0.2, fork_probability=0.2):
        super().__init__()
        self.x = x
        self.operators = operators
        self.command_probability = command_probability
        self.node_probability = node_probability
        self.parameter_probability = parameter_probability
        self.prune_probability = prune_probability
        self.fork_probability = fork_probability

    def set_parameters(self, para_dict):
        self.x = para_dict["x"]
        self.command_probability = para_dict["command_probability"]
        self.node_probability = para_dict["node_probability"]
        self.parameter_probability = para_dict["parameter_probability"]
        self.prune_probability = para_dict["prune_probability"]
        self.fork_probability = para_dict["fork_probability"]

    def do(self, population):
        component_generator = ComponentGenerator(self.x.shape[1])
        for op in self.operators:
            component_generator.add_operator(op)
        mutation = AGraphMutation(component_generator, self.command_probability
                                  , self.node_probability, self.parameter_probability
                                  , self.prune_probability, self.fork_probability)
        population.set_pop_size(len(population.pop_list))
        parent_num = np.random.randint(low=0, high=population.get_pop_size() - 1)
        parent = population.pop_list[parent_num]
        bingo_parent = AGraph(equation=str(parent.equation))
        bingo_parent._update()
        bingo_child = mutation(bingo_parent)

        child = Individual(str(bingo_child))
        population.append(child)
        new_pop_size = population.get_pop_size() + 1
        population.set_pop_size(new_pop_size)
