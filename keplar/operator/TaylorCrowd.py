
from keplar.operator.operator import Operator


class TaylorCrowd(Operator):
    def __init__(self):
        pass

    def do(self,population,front,reminder):
        cur_population = copy.deepcopy([population[i] for i in front])
        cur_population.sort(key=lambda x: x.raw_fitness_)
        sorted1 = copy.deepcopy(cur_population)
        cur_population.sort(key=lambda x: x.length_)
        sorted2 = cur_population
        distance = [0 for i in range(0, len(front))]
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        fitness_ = [sorted1[i].raw_fitness_ for i in range(len(cur_population))]
        length_ =  [sorted2[i].length_ for i in range(len(cur_population))]
        maxFit,minFit,maxLen,minLen = max(fitness_),min(fitness_),max(length_),min(length_)
        #第k个个体的距离就是front[k]的距离----dis[k]==front[k]
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (fitness_[k + 1] - fitness_[k - 1]) / (
                        maxFit - minFit+0.01)
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (length_[k + 1] - length_[k - 1]) / (
                        maxLen - minLen+0.01)
        index_ = sorted(range(len(distance)),key=lambda k:distance[k])
        index_.reverse()
        reminderPop = [cur_population[i] for i in index_][:reminder]
        return reminderPop