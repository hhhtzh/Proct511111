
from keplar.operator.operator import Operator


class TaylorSort(Operator):
    def __init__(self):
        pass

    def do(self,population):
        programs = population.target_pop_list
        S = [[] for i in range(0, len(programs))]
        front = [[]]
        n = [0 for i in range(0, len(programs))]
        rank = [0 for i in range(0, len(programs))]
        # 计算种群中每个个体的两个参数 n[p]和 S[p] ; 并将种群中参数n[p]=0的个体索引放入集合F1中
        for p in range(0, len(programs)):
            S[p] = []
            n[p] = 0
            for q in range(0, len(programs)):
                # if p domains q:
                if (programs[p].length_ < programs[q].length_ and programs[p].raw_fitness_ < programs[q].raw_fitness_) or (
                        programs[p].length_ <= programs[q].length_ and programs[p].raw_fitness_ < programs[q].raw_fitness_) or (
                        programs[p].length_ < programs[q].length_ and programs[p].raw_fitness_ <= programs[q].raw_fitness_):
                    if q not in S[p]:
                        S[p].append(q)
                elif (programs[q].length_ < programs[p].length_ and programs[q].raw_fitness_ < programs[p].raw_fitness_) or (
                        programs[q].length_ <= programs[p].length_ and programs[q].raw_fitness_ < programs[p].raw_fitness_) or (
                        programs[q].length_ < programs[p].length_ and programs[q].raw_fitness_ <= programs[p].raw_fitness_):
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)
        # 计算其他非帕累托前沿个体的等级并存入集合，并使用rank记录排名等级
        i = 0
        while (front[i] != []):
            Q = []
            # print(type(front[i]))
            for p in iter(front[i]):
                for q in iter(S[p]):
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            front.append(Q)

        del front[len(front) - 1]
        # print(front)
        return front