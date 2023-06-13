import random

from bingo.symbolic_regression import ComponentGenerator, AGraph
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from keplar.population.individual import Individual
from keplar.operator.operator import Operator
import numpy as np
# import pyoperon as Operon

from keplar.translator.translator import to_op


class Mutation(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        pass


class BingoMutation(Mutation):
    def __init__(self, x, operators, to_type, command_probability=0.2, node_probability=0.2, parameter_probability=0.2,
                 prune_probability=0.2, fork_probability=0.2):
        super().__init__()
        self.to_type = to_type
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
        if population.pop_type != "Bingo":
            population.set_pop_size(len(population.target_pop_list))
            parent_num = np.random.randint(low=0, high=population.get_pop_size() - 1)
            bingo_parent = population.target_pop_list[parent_num]
            bingo_parent._update()
            bingo_child = mutation(bingo_parent)
        else:
            population.set_pop_size(len(population.target_pop_list))
            parent_num = np.random.randint(low=0, high=population.get_pop_size() - 1)
            parent = population.target_pop_list[parent_num]
            # parent._update()
            bingo_child = mutation(parent)
        if self.to_type != "Bingo":
            child = Individual(str(bingo_child))
            population.append(child)
            new_pop_size = population.get_pop_size() + 1
            population.set_pop_size(new_pop_size)
        else:
            population.pop_type = "Bingo"
            population.self_pop_enable = False
            population.target_pop_list.append(bingo_child)
            new_pop_size = population.get_pop_size() + 1
            population.set_pop_size(new_pop_size)


class OperonMutation(Mutation):
    def __init__(self, onepoint_p, changevar_p, changefunc_p, replace_p, np_x, np_y, maxD, maxL, tree_type, to_type):
        super().__init__()
        self.to_type = to_type
        if tree_type == "balanced" or tree_type == "probabilistic":
            self.tree_type = tree_type
        else:
            raise ValueError("创建树的类型错误")
        self.replace_p = replace_p
        self.changefunc_p = changefunc_p
        self.changevar_p = changevar_p
        self.onepoint_p = onepoint_p
        self.maxL = maxL
        self.maxD = maxD
        self.np_x=np_x
        np_y = np_y.reshape([-1, 1])
        self.np_y = np_y
        self.ds = Operon.Dataset(np.hstack([np_x, np_y]))

    def do(self, population):
        target = self.ds.Variables[-1]
        inputs = Operon.VariableCollection(v for v in self.ds.Variables if v.Name != target.Name)
        pset = Operon.PrimitiveSet()
        pset.SetConfig(
            Operon.PrimitiveSet.TypeCoherent)
        if self.tree_type == "balanced":
            tree_creator = Operon.BalancedTreeCreator(pset, inputs, bias=0.0)
        elif self.tree_type == "probabilistic":
            tree_creator = Operon.ProbabilisticTreeCreator(pset, inputs, bias=0.0)
        else:
            raise ValueError("Operon创建树的类型名称错误")

        coeff_initializer = Operon.NormalCoefficientInitializer()
        coeff_initializer.ParameterizeDistribution(0, 1)
        mut_onepoint = Operon.NormalOnePointMutation()
        mut_changeVar = Operon.ChangeVariableMutation(inputs)
        mut_changeFunc = Operon.ChangeFunctionMutation(pset)
        mut_replace = Operon.ReplaceSubtreeMutation(tree_creator, coeff_initializer, self.maxD, self.maxL)
        mutation = Operon.MultiMutation()
        mutation.Add(mut_onepoint, self.onepoint_p)
        mutation.Add(mut_changeVar, self.changevar_p)
        mutation.Add(mut_changeFunc, self.changefunc_p)
        mutation.Add(mut_replace, self.replace_p)
        rng = Operon.RomuTrio(random.randint(1, 1000000))
        if population.pop_type == "Operon":
            new_tree_list = []
            for i in population.target_pop_list:
                new_tree_list.append(mutation(rng, i))
            population.target_pop_list = new_tree_list
            population.set_pop_size(len(new_tree_list))
            if self.to_type == "Operon":
                population.self_pop_enable = False

            else:
                pass
        else:
            new_tree_list = []
            for i in population.pop_list:
                op_tree=to_op(i,self.np_x,self.np_y)
                new_tree_list.append(mutation(rng, op_tree))
                population.target_pop_list = new_tree_list
                population.set_pop_size(len(new_tree_list))
                if self.to_type == "Operon":
                    population.self_pop_enable = False
                else:
                    pass


