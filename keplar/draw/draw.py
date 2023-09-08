class Draw:
    def __init__(self, types, file_path, names):
        self.names = names
        self.file_path = file_path
        self.types = types

    def draw(self):
        raise NotImplementedError


class BoxImgDraw(Draw):
    def __init__(self, types, file_path, names):
        super().__init__(types, file_path, names)

    def draw(self):
        if self.types=="csv":

