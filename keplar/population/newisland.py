import numpy as np

from bingo.evolutionary_optimizers.island import Island


class NewIsland(Island):
    def __init__(self, evolution_algorithm, generator, population_size):
        super().__init__(evolution_algorithm, generator, population_size)

    def get3top(self):
        self.evaluate_population()
        best3 = [self.population[0]]
        for indv in self.population:
            if len(best3) == 1:
                if indv.fitness == best3[0].fitness:
                    continue
                best3.append(indv)
            elif len(best3) == 2:
                if indv.fitness == best3[0].fitness or indv.fitness == best3[1].fitness:
                    continue
                best3.append(indv)
            else:
                if indv.fitness < best3[2].fitness or np.isnan(best3[2].fitness).any():
                    best3[2] = indv
                    for i in range(3):
                        for j in range(3):
                            if i < j and best3[i].fitness > best3[j].fitness:
                                temp_indv = best3[i]
                                best3[i] = best3[j]
                                best3[j] = temp_indv
        return best3
