class Operator:
    def __init__(self):
        self.type = None

    def do(self, population):
        pass

    def pre_do(self):
        pass

    def pos_do(self):
        pass

    def exec(self, population):
        self.pre_do()
        self.do(population)
        self.pos_do()

    def set_parameters(self, para_dict):
        pass

    def __init_config__(self, config):
        pass