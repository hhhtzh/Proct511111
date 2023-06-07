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

    def get_pop_size(self):
        return self.pop_size

    def set_pop_size(self, new_pop_size):
        self.pop_size = new_pop_size
