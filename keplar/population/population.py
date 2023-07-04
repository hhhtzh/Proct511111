
class Population:
    def __init__(self, pop_size):
        self.pop_list = []
        self.pop_size = pop_size
        self.pop_type = None
        self.self_pop_enable = None
        self.translate_flag = True
        self.target_pop_list = []
        
        self.target_fit_list=[]

    def initial(self, pop_list):
        self.pop_list = pop_list

    def check_flag(self, to_type):
        if self.pop_type == to_type:
            self.translate_flag = False
        return self.translate_flag

    def append(self, ind):
        self.pop_list.append(ind)
        self.pop_size += 1

    def target_append(self,ind):
        self.target_pop_list.append(ind)
        self.pop_size +=1

    def get_pop_size(self):
        if self.pop_type=="self":
            self.pop_size=len(self.pop_list)
        else:
            self.pop_size=len(self.target_pop_list)
        return self.pop_size

    def set_pop_size(self, new_pop_size):
        self.pop_size = new_pop_size

    def get_tar_best(self):
        best_fitness = self.target_fit_list[0]
        best_index=0
        for j in range(len(self.target_fit_list)):
            if self.target_fit_list[j] < best_fitness:
                best_fitness = self.target_fit_list[j]
                best_index=j
        return best_index

    def get_best(self):
        best_index = 0
        best_fitness = self.pop_list[0].get_fitness()
        for j in range(len(self.pop_list)):
            if self.pop_list[j].get_fitness() < best_fitness:
                best_fitness = self.pop_list[j].get_fitness()
                best_index = j
        return best_index

    def get_best_fitness(self):
        if self.pop_type != "self":
            best_num = self.get_tar_best()
            return self.target_fit_list[best_num]
        else:
            best_num = self.get_best()
            return self.pop_list[best_num].get_fitness()


