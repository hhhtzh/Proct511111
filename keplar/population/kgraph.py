from keplar.population.population import Population


class KeplarGraphPopulation(Population):
    def __init__(self, pop_size):
        self.pop_graph = None
        self.inner_pop = None
        if pop_size != "8x8" or "16x16":
            raise ValueError("请输入正确格式的种群方阵规模")
        super().__init__(pop_size)

    def initial(self, normal_pop):
        if normal_pop is not isinstance(normal_pop, Population):
            raise ValueError("内部种群必须为Keplar种群类型")
        self.inner_pop = normal_pop
        if self.pop_size == "8x8":
            if self.inner_pop.get_pop_size() < 64:
                raise ValueError("个体数量不够")
            pop_index = 0
            self.pop_graph = []
            for i in range(8):
                temp_list = []
                for j in range(8):
                    temp_list.append(self.inner_pop.pop_list[pop_index])
                    pop_index += 1
                self.pop_graph.append(temp_list)
        elif self.pop_graph == "16x16":
            if self.inner_pop.get_pop_size() < 256:
                raise ValueError("个体数量不够")
            pop_index = 0
            self.pop_graph = []
            for i in range(16):
                temp_list = []
                for j in range(16):
                    temp_list.append(self.inner_pop.pop_list[pop_index])
                    pop_index += 1
                self.pop_graph.append(temp_list)
        else:
            raise ValueError("输入规模不支持")

    def update(self):
        if self.pop_graph == "8x8":
            for i in range(8):
                for j in range(8):
                    for k in range(8):
                        if self.pop_graph[i][j].fitness < self.pop_graph[i][k].fitness:
                            temp = self.pop_graph[i][j]
                            self.pop_graph[i][j] = self.pop_graph[i][k]
                            self.pop_graph[i][k] = temp

        elif self.pop_graph == "16x16":
            for i in range(16):
                for j in range(16):
                    for k in range(16):
                        if self.pop_graph[i][j].fitness < self.pop_graph[i][k].fitness:
                            temp = self.pop_graph[i][j]
                            self.pop_graph[i][j] = self.pop_graph[i][k]
                            self.pop_graph[i][k] = temp


    def out_put(self):
        
