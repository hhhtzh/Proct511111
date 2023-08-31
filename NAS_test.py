import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch import Tensor

from keplar.data.data import Data


# 定义一个简单的神经网络结构作为评估器
class PerformancePredictor(nn.Module):
    def __init__(self, input_size):
        super(PerformancePredictor, self).__init__()
        self.fc = nn.Linear(input_size, 3)  # 预测每个演化算法的性能
        self.softmax = nn.Softmax(dim=1)  # 在输出层使用softmax函数

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


# 定义演化算法的选择器
def select_algorithm(predictor, input_data):
    with torch.no_grad():
        algorithm_probabilities = predictor(input_data)
        selected_algorithm = torch.argmax(algorithm_probabilities).item()
    return selected_algorithm


# 模拟训练数据和演化算法选择
data = pd.read_csv("NAStraining_data/recursion_training2.csv")
y = data['SelectedLable']  # label值
list1 = y.to_list()
arr = np.array(list1, dtype=float)
x = data.drop(['SelectedLable'], axis=1)
x = np.array(x, dtype=float)
print(np.shape(y))
algorithm_labels = np.clip(arr, 0, 2)  # 将值限制在0到2之间
algorithm_labels = torch.from_numpy(algorithm_labels).long()  # 转换为长整型

training_data = torch.from_numpy(x).float()  # 特征数量需要根据实际情况调整
# algorithm_labels = torch.from_numpy(arr).float()  # 演化算法的实际选择
# training_data =Tensor( data.iloc[:,:-2])# 特征数量需要根据实际情况调整
print(y)
# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
# data.read_file()
# dataSet = data.get_np_ds()
# training_data = torch.tensor(dataSet)
# training_data=training_data.to(torch.float32)
# print(np.shape(dataSet))
# algorithm_labels = torch.randint(3, (1000,))  # 演化算法的实际选择
# algorithm_labels = Tensor(data.iloc[:,-1])# 演化算法的实际选择
# algorithm_labels=algorithm_labels.to(torch.float32)
# 创建可微神经网络实例
predictor = PerformancePredictor(input_size=7)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(predictor.parameters(), lr=0.001)

# 训练神经网络
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted_probabilities = predictor(training_data)
    predicted_probabilities = predicted_probabilities.type(torch.float32)
    algorithm_labels = algorithm_labels.type(torch.long)
    loss = criterion(predicted_probabilities, algorithm_labels)
    loss.backward()
    optimizer.step()

print("Training Finished!")

# 在每一代中选择演化算法
for generation in range(1000):
    selected_algorithm = select_algorithm(predictor, training_data[generation].unsqueeze(0))
    print(f"Generation {generation + 1}: Selected Algorithm {selected_algorithm + 1}")
