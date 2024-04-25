import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms
from tqdm import tqdm
import copy
import numpy as np
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt


device = torch.device("mps")


class BatchBernoulliSampler:
    """
    实现有放回的伯努利抽样，每个样本可以被考虑多次，但每个epoch的迭代次数是固定的。
    """

    def __init__(self, data_source, batch_size = 16, p=0.8, num_iterations=1000):
        self.data_source = data_source
        self.batch_size = batch_size
        self.p = p
        self.num_iterations = num_iterations if num_iterations is not None else len(data_source) // batch_size

    def __iter__(self):
        n = len(self.data_source)

        for _ in range(self.num_iterations):
            batch_indices = torch.randint(0, n, (self.batch_size,))
            # 对这些样本进行伯努利抽样
            final_batch = [idx.item() for idx in batch_indices if torch.rand(1).item() < self.p]
            if len(final_batch) > 0:
                yield final_batch

    def __len__(self):
        # 总的迭代次数是固定的
        return self.num_iterations



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_data(transform):
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
    sampler = BatchBernoulliSampler(trainset)
    train_loader = DataLoader(trainset, batch_sampler=sampler)
    return train_loader


class CovNet(nn.Module):
    def __init__(self):
        super(CovNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_svrg(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()
    losses = []
    log_interval = 100  # 每10个批次记录一次损失

    # Step 1: 计算全数据的梯度，称为泰勒梯度
    full_grad = None
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if full_grad is None:
            full_grad = [p.grad.data.clone() for p in model.parameters()]
        else:
            for idx, param in enumerate(model.parameters()):
                full_grad[idx] += param.grad.data.clone()

    for param, grad in zip(model.parameters(), full_grad):
        grad /= len(train_loader)

    # Step 2: 使用 SVRG 更新
    batch_count = 0
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            with torch.no_grad():
                for param, full_grad_param in zip(model.parameters(), full_grad):
                    param.grad.data.sub_(full_grad_param - param.grad.data)

            optimizer.step()
            batch_count += 1
            if batch_count % log_interval == 0:
                losses.append(loss.item())

            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}], Loss: {loss.item():.4f}')

    return losses

def moving_average(losses, window_size):
    """计算给定窗口大小的移动平均值"""
    cumsum_vec = np.cumsum(np.insert(losses, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

train_loader = load_data(transform=transform)
model = CovNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 300

losses = train_svrg(model, train_loader, optimizer, criterion, num_epochs, device)

window_size = 10
smoothed_losses = moving_average(losses, window_size)

plt.figure(figsize=(10, 5))
plt.plot(smoothed_losses, marker='o', linestyle='-', markersize=4)
plt.xlabel(f'Number of Iterations (x100 batches)')
plt.ylabel('Smoothed Loss')
plt.title('Smoothed Training Loss over Iterations')
plt.savefig('Smoothed_Training_Loss.png')
plt.show()


