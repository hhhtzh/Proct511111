class Population:
    def __init__(self, pop_size):
        self.pop_list = []
        self.pop_size = pop_size

    def initial(self, pop_list):
        self.pop_list = pop_list

    def append(self, ind):
        self.pop_list.append(ind)
        self.pop_size += 1

    def get_pop_size(self):
        return self.pop_size

    def set_pop_size(self, new_pop_size):
        self.pop_size = new_pop_size


        现象学现象学
