import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import KFold
from data_loader import load_dataset
from network import simpleVGG
import wandb
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
modle = simpleVGG()

# hyper-params
num_epochs = 100
k_folds = 3
batch_size = 32
lr = 1e-3
use_wandb = False

train_data = load_dataset("rps")
test_data = load_dataset("rps-test-set")

# 交叉验证调参
kf = KFold(n_splits=k_folds, shuffle=True, random_state=2026)

fold_results = []
train_data_idx = range(len(train_data))
for fold, (train_index, val_index) in enumerate(kf.split(train_data_idx)):
    print(f"Training Fold {fold + 1}/{k_folds}")

    # 根据split结果创建子数据集
    train_subset = Subset(train_data, train_index)
    val_subset = Subset(train_data, val_index)
    trian_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size)

    # 每个fold新建一个模型
    model = simpleVGG().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 训练主循环
    for epoch in range(num_epochs):
        model.train()
        for images, labels in trian_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss)

    # 计算验证集准确率
    model.eval()
    correct_num = 0
    total_num = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              _, pred_label = torch.max(outputs.detach(), 1)
              total_num += labels.size(0)
              correct_num += (pred_label == labels).sum().item()

    accuracy = 100 * correct_num / total_num
    fold_results.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")

# 输出最终结果
print(f"\nMean Accuracy: {np.mean(fold_results):.2f}% (+/- {np.std(fold_results):.2f}%)")

# 根据调参结果使用全部训练集重新训练一个模型,并在测试集上测试准确率
