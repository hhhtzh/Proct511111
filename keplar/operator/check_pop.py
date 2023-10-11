import numpy as np

from keplar.operator.operator import Operator


class CheckPopulation(Operator):
    def __init__(self, data=None):
        super().__init__()
        self.data = data
        self.generation=0

    def do(self, population):
        length_list = []
        fit_list = []
        if population.pop_type != "self":
            raise ValueError("暂时不支持")
        else:
            # print(len(population.pop_list))
            for ind in population.pop_list:
                length_list.append(len(ind.func))
                fit_list.append(ind.fitness)
                # print(ind.format())
                # print(ind.fitness)
            # print(fit_list)
            np_length = np.array(length_list)

            np_fit = np.array(fit_list)
            print(np_length)
            # print(np_fit)
            max_length = np.max(np_length)
            min_length = np.min(np_length)
            mean_length = np.mean(np_length)
            best_fit = np.min(np_fit)
            worest_fit = np.max(np_fit)
            mean_fit = np.mean(np_fit)
            return [best_fit, worest_fit, mean_fit, max_length, min_length, mean_length]

    def write_jason(self,population,file_name):
        length_list = []
        fit_list = []
        if population.pop_type != "self":
            raise ValueError("暂时不支持")
        else:
            # print(len(population.pop_list))
            for ind in population.pop_list:
                length_list.append(len(ind.func))
                fit_list.append(ind.fitness)
                # print(ind.format())
                # print(ind.fitness)
            # print(fit_list)
            np_length = np.array(length_list)

            np_fit = np.array(fit_list)
            print(np_length)
            # print(np_fit)
            max_length = np.max(np_length)
            min_length = np.min(np_length)
            mean_length = np.mean(np_length)
            best_fit = np.min(np_fit)
            min_index = np.argmin(np_fit)
            best_ind = str(population.pop_list[min_index].format())
            worest_fit = np.max(np_fit)
            mean_fit = np.mean(np_fit)
        state_list=[best_fit, worest_fit, mean_fit, max_length, min_length, mean_length,best_ind]
        self.generation+=1
        data = {
            "generation":self.generation,
            "best_fit": state_list[0],
            "worest_fit": state_list[1],
            "mean_fit": state_list[2],
            "best_ind":state_list[6],
            "max_length":state_list[3],
            "min_length":state_list[4],
            "mean_length":state_list[5],
        }
        # 指定要写入的 JSON 文件路径
        file_path = "data.json"

        # 使用 with 语句打开文件，以写入模式打开
        with open(file_path, "w") as json_file:
            # 使用 json.dump() 将数据写入文件
            json.dump(data, json_file)

        print("Data has been written to", file_path)
        if

