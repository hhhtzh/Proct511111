import torch
import torch.nn as nn
import torch.optim as optim
import random

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

# 定义演化算法的选择函数
def select_algorithm(predictor, input_data):
    with torch.no_grad():
        algorithm_probabilities = predictor(input_data)
        selected_algorithm = torch.argmax(algorithm_probabilities).item()
    return selected_algorithm

# 模拟训练数据和演化算法选择
num_generations = 1000  # 修改为你的数据集行数
input_size = 6  # 修改为你的输入特征维度
training_data = torch.randn(num_generations, input_size)  # 数据集大小为 (num_generations, input_size)
# data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
# data.read_file()
# dataSet = data.get_np_ds()
# training_data = torch.tensor(dataSet)
# training_data=training_data.to(tensor.f)
algorithm_labels = torch.randint(3, (num_generations,))  # 演化算法的实际选择

# 创建可微神经网络实例
predictor = PerformancePredictor(input_size=input_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(predictor.parameters(), lr=0.001)

# 训练神经网络
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted_probabilities = predictor(training_data)
    loss = criterion(predicted_probabilities, algorithm_labels)
    loss.backward()
    optimizer.step()

print("Training Finished!")

# 在每一代中选择演化算法
for generation in range(num_generations):
    print(training_data[generation].unsqueeze(0))
    selected_algorithm = select_algorithm(predictor, training_data[generation].unsqueeze(0))
    print(f"Generation {generation+1}: Selected Algorithm {selected_algorithm+1}")

