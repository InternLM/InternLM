import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from apex import amp
import torch.nn.functional as F

# 生成示例数据
np.random.seed(42)
X = np.random.rand(100, 5)
y = np.random.randint(2, size=100)

# 将数据分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X, dtype=torch.float16).to('cuda')
y_train_tensor = torch.tensor(y, dtype=torch.long).to('cuda')
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    # @amp.half_function
    def forward(self, input):
        with torch.autocast('cuda', enabled=False):
            # import pdb; pdb.set_trace()
            output = torch.matmul(input, self.weight.t())
            if self.bias is not None:
                output += self.bias
        return output

# 定义自定义神经网络模型
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomModel, self).__init__()
        # self.fc1 = CustomLinear(input_size, hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        # self.fc2 = CustomLinear(hidden_size, num_classes)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        import pdb; pdb.set_trace()
        out = self.fc1(x)
        out = self.relu(out)
        # with torch.cuda.amp.autocast(dtype=torch.float32):
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=1)  # 添加 softmax 操作
        return out

# 初始化模型
input_size = X_train_tensor.shape[1]
hidden_size = 16
num_classes = 2
model = CustomModel(input_size, hidden_size, num_classes).to('cuda')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

# 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # import pdb; pdb.set_trace()
    with torch.cuda.amp.autocast(dtype=torch.float16):
    # with torch.autocast('cuda', dtype=torch.float16):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
    loss.backward()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    
    print(outputs.dtype)

print("Done!")

# 在测试集上进行预测
# with torch.no_grad():
#     test_outputs = model(X_test_tensor)
#     predictions = np.argmax(test_outputs, axis=1)

# 计算准确率
# accuracy = accuracy_score(y_test, predictions)
# print(f"准确率：{accuracy:.2f}")

# 获取当前分配的显存（in bytes）
allocated_memory = torch.cuda.memory_allocated(device='cuda') / 1024**2  # 转换为MB
print("Allocated Memory:", allocated_memory, "MB")

# 获取分配的最大显存（in bytes）
max_allocated_memory = torch.cuda.max_memory_allocated(device='cuda') / 1024**2  # 转换为MB
print("Max Allocated Memory:", max_allocated_memory, "MB")
