import json
import logging
import os

import numpy as np

from keplar.operator.operator import Operator


class CheckPopulation(Operator):
    def __init__(self, data=None):
        super().__init__()
        self.data = data
        self.generation = 0
        self.history_best_ind = None
        self.history_best_fit = None

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

    def write_rl_json(self, population, action_seq, file_name):
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
            actions_length = len(action_seq)
            if self.history_best_ind is None:
                self.history_best_fit = best_fit
                self.history_best_ind = best_ind
            if best_fit < self.history_best_fit:
                self.history_best_fit = best_fit
                self.history_best_ind = best_ind

        state_list = [best_fit, worest_fit, mean_fit, max_length, min_length, mean_length, best_ind]
        self.generation += 1
        data = {
            "generation": int(self.generation),
            "best_fit": float(state_list[0]),
            "worest_fit": float(state_list[1]),
            "mean_fit": float(state_list[2]),
            "best_ind": state_list[6],
            "history_best_fit": float(self.history_best_fit),
            "history_best_ind": self.history_best_ind,
            "action_length": int(actions_length),
            "actions": str(action_seq),
            "max_length": int(state_list[3]),
            "min_length": int(state_list[4]),
            "mean_length": state_list[5],
        }
        # 指定要写入的 JSON 文件路径
        file_path = "result/" + str(file_name) + ".json"

        # 使用 with 语句打开文件，以写入模式打开
        def is_json_empty(file_path):
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                return not bool(data)  # 如果文件为空，返回True；否则返回False

            except (FileNotFoundError, json.decoder.JSONDecodeError):
                return True  # 文件不存在或无法解析为JSON时，也视为空

        # 创建或更新 JSON 文件
        def create_or_update_json(file_path, outer_key, data):
            if is_json_empty(file_path):
                # 如果文件为空，创建一个包含标签的 JSON
                with open(file_path, "w") as file:
                    json.dump({outer_key: data}, file, indent=4)
            else:
                # 如果文件不为空，读取原有数据，添加新数据，并写回文件
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
                existing_data[outer_key] = data
                with open(file_path, "w") as file:
                    json.dump(existing_data, file, indent=4)

        # # 检查文件是否存在
        # if not os.path.exists(file_path):
        #     # 如果文件不存在，创建一个空的JSON文件
        #     with open(file_path, 'w') as file:
        #         json.dump({}, file)

        # 现在你可以继续操作该文件

        # 外层标签和要写入的数据
        outer_key = "data_gen" + str(self.generation)
        data_to_write = data

        # 调用函数创建或更新 JSON 文件
        create_or_update_json(file_path, outer_key, data_to_write)

    def write_jason(self, population, file_name):
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
            if self.history_best_ind is None:
                self.history_best_fit = best_fit
                self.history_best_ind = best_ind
            if best_fit < self.history_best_fit:
                self.history_best_fit = best_fit
                self.history_best_ind = best_ind

        state_list = [best_fit, worest_fit, mean_fit, max_length, min_length, mean_length, best_ind]
        self.generation += 1
        data = {
            "generation": int(self.generation),
            "best_fit": float(state_list[0]),
            "worest_fit": float(state_list[1]),
            "mean_fit": float(state_list[2]),
            "best_ind": state_list[6],
            "history_best_fit": float(self.history_best_fit),
            "history_best_ind": self.history_best_ind,
            "max_length": int(state_list[3]),
            "min_length": int(state_list[4]),
            "mean_length": state_list[5],
        }
        # 指定要写入的 JSON 文件路径
        file_path = "result/" + str(file_name) + ".json"

        # 使用 with 语句打开文件，以写入模式打开
        def is_json_empty(file_path):
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                return not bool(data)  # 如果文件为空，返回True；否则返回False
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                return True  # 文件不存在或无法解析为JSON时，也视为空

        # 创建或更新 JSON 文件
        def create_or_update_json(file_path, outer_key, data):
            if is_json_empty(file_path):
                # 如果文件为空，创建一个包含标签的 JSON
                with open(file_path, "w") as file:
                    json.dump({outer_key: data}, file, indent=4)
            else:
                # 如果文件不为空，读取原有数据，添加新数据，并写回文件
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
                existing_data[outer_key] = data
                with open(file_path, "w") as file:
                    json.dump(existing_data, file, indent=4)

        # # 检查文件是否存在
        # if not os.path.exists(file_path):
        #     # 如果文件不存在，创建一个空的JSON文件
        #     with open(file_path, 'w') as file:
        #         json.dump({}, file)

        # 现在你可以继续操作该文件

        # 外层标签和要写入的数据
        outer_key = "data_gen" + str(self.generation)
        data_to_write = data

        # 调用函数创建或更新 JSON 文件
        create_or_update_json(file_path, outer_key, data_to_write)


    def write_log(self,set_level):
        logger = logging.getLogger('keplar_logger')

