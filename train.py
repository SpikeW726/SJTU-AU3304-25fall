import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from data_loader import load_dataset
from network import simpleVGG
import wandb
import numpy as np
from datetime import datetime
import os
import random
import math
import json
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy

class TransformSubset(torch.utils.data.Dataset):
    """为了实现同一个Dataset实例划分后采取不同图像处理,需要定义一个包装类"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, label

def train(device, train_dataset, val_dataset=None, test_dataset=None,
          num_epochs=10, batch_size=32, lr=1e-3,
          model_path:str="", use_wandb=False, patience=50, min_delta=1e-4):
    """
    使用传入的超参和数据集进行训练,保存并返回最终模型
    如果use_wandb=True,则使用wandb记录loss曲线,否则仅在终端输出

    Args:
        patience: 早停等待的epoch数
        min_delta: 认为有改善的最小变化量
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = simpleVGG().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 用于记录历史数据
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    # Early stopping 和 best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # 训练主循环
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Validation phase - 复用 evaluate 函数
        val_loss = 0.0
        val_acc = 0.0
        if val_dataset is not None:
            val_acc, val_loss = evaluate(model, val_dataset, batch_size, device,
                                         return_loss=True, loss_fn=loss_fn, verbose=False)

        # Log metrics after each epoch
        if val_dataset is not None:
            print(f"Epoch {epoch+1}/{num_epochs}, train_loss={avg_train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, val_acc={val_acc:.2f}%")

            # 记录历史数据
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Early stopping 和 best model tracking
            if val_loss < best_val_loss - min_delta:
                # 有显著改善，更新最佳模型
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"  ✓ New best model! val_loss: {val_loss:.6f}")
            else:
                # 无显著改善，增加patience计数器
                patience_counter += 1
                print(f"  ✗ No improvement. Patience: {patience_counter}/{patience}")

                # 检查是否需要早停
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                    print(f"Best val_loss: {best_val_loss:.6f}")
                    break

            if use_wandb:
                # 记录数值到 wandb
                wandb.log({
                    "loss/train": avg_train_loss,
                    "loss/val": val_loss,
                    "accuracy/val": val_acc
                }, step=epoch + 1)

                # 创建合并的 loss 曲线图
                fig, ax = plt.subplots(figsize=(10, 6))
                epochs = range(1, epoch + 2)
                ax.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
                ax.plot(epochs, history["val_loss"], label="Val Loss", color="red")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training and Validation Loss")
                ax.legend()
                ax.grid(True)

                # 上传到 wandb
                wandb.log({"loss_curve": wandb.Image(fig)}, step=epoch + 1)
                plt.close(fig)
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, train_loss={avg_train_loss:.6f}")
            if use_wandb:
                wandb.log({"loss/train": avg_train_loss}, step=epoch + 1)

    # 训练循环结束后，保存最终的latest模型状态
    # 无论训练是早停结束还是正常结束，这里都保存训练结束时的模型
    latest_model_state = copy.deepcopy(model.state_dict())
    print(f"\n[DEBUG] Saved final latest model state after training loop ends")

    # 训练结束后，分别评估最佳模型和最新模型
    print(f"\n[DEBUG] test_dataset is not None: {test_dataset is not None}")
    print(f"[DEBUG] best_model_state is not None: {best_model_state is not None}")
    print(f"[DEBUG] latest_model_state is not None: {latest_model_state is not None}")

    if test_dataset is not None and best_model_state is not None and latest_model_state is not None:
        print("\n" + "="*60)
        print("Training completed! Evaluating both models on test set...")
        print("="*60)

        # 评估最佳模型
        print("\n[1/2] Evaluating BEST model (based on validation loss)...")
        print(f"[DEBUG] Best model's val_loss: {best_val_loss:.6f}")
        print("-"*60)
        model.load_state_dict(best_model_state)
        best_test_acc, best_test_loss = evaluate(model, test_dataset, batch_size, device,
                                                  return_loss=True, loss_fn=loss_fn, verbose=True)

        # 评估最新模型
        print("\n[2/2] Evaluating LATEST model (at training end)...")
        print("-"*60)
        model.load_state_dict(latest_model_state)
        latest_test_acc, latest_test_loss = evaluate(model, test_dataset, batch_size, device,
                                                     return_loss=True, loss_fn=loss_fn, verbose=True)

        # 对比结果
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Best Model  - Test Loss: {best_test_loss:.6f}, Test Acc: {best_test_acc:.2f}%")
        print(f"Latest Model - Test Loss: {latest_test_loss:.6f}, Test Acc: {latest_test_acc:.2f}%")

        if best_test_acc > latest_test_acc:
            diff = best_test_acc - latest_test_acc
            print(f"✓ Best model performs better by {diff:.2f}%")
        elif best_test_acc < latest_test_acc:
            diff = latest_test_acc - best_test_acc
            print(f"✗ Latest model performs better by {diff:.2f}%")
        else:
            print("= Both models have equal accuracy")
        print("="*60 + "\n")

        # 保存最佳模型
        if model_path != "":
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(best_model_state, model_path)
            print(f"Best model saved to: {model_path}\n")

        if use_wandb:
            wandb.log({
                "test_best/accuracy": best_test_acc,
                "test_best/loss": best_test_loss,
                "test_latest/accuracy": latest_test_acc,
                "test_latest/loss": latest_test_loss
            })
    elif test_dataset is not None:
        # 如果只有best model（正常情况不应该发生）
        print("\n" + "="*60)
        print("Training completed! Evaluating on test set...")
        print("="*60)
        model.load_state_dict(best_model_state)
        test_acc, test_loss = evaluate(model, test_dataset, batch_size, device,
                                       return_loss=True, loss_fn=loss_fn, verbose=True)
        print("="*60 + "\n")

        if use_wandb:
            wandb.log({
                "test/accuracy": test_acc,
                "test/loss": test_loss
            })

        if model_path != "":
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    return model

def evaluate(model, dataset, batch_size, device, return_loss=False, loss_fn=None, verbose=True):
    """
    使用传入的模型在指定数据集上进行评估

    Args:
        return_loss: 是否返回loss
        loss_fn: 如果return_loss=True,需要传入loss函数
        verbose: 是否打印结果

    Returns:
        如果return_loss=False: 返回accuracy
        如果return_loss=True: 返回 (accuracy, loss)
    """
    model.eval()
    eval_dataloader = DataLoader(dataset, batch_size=batch_size)
    correct_num = 0
    total_num = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in eval_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if return_loss and loss_fn is not None:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()

            _, pred_label = torch.max(outputs.detach(), 1)
            total_num += labels.size(0)
            correct_num += (pred_label == labels).sum().item()

    accuracy = 100 * correct_num / total_num

    if return_loss:
        avg_loss = total_loss / len(eval_dataloader)
        if verbose:
            print(f"Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss
    else:
        if verbose:
            print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

if __name__ == "__main__":
    # hyper-params
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    num_epochs = 300
    batch_size = 32
    lr = 1e-3
    use_wandb = True
    train_val_split = 0.8
    patience = 50             
    min_delta = 1e-5
    random_seed = 42           

    # 生成模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("results\models5", f"train_model_{timestamp}.pth")

    # 导入训练/测试数据集
    train_data = load_dataset("rps")
    test_data = load_dataset("rps-test-set")

    # wandb启动配置
    if use_wandb:
        run_name = f"simpleVGG_lr{lr}_bs{batch_size}_e{num_epochs}_{timestamp}"
        wandb.init(
            project="rps-classification",
            name=run_name,
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "train_val_split": train_val_split,
                "patience": patience,
                "min_delta": min_delta,
                "random_seed": random_seed,
            }
        )

    # 按比例分出训练集和验证集,注意保持各类型样本数保持原比例,即采用“分层划分”
    labels = [train_data.samples[i][1] for i in range(len(train_data))]
    train_idx, val_idx = train_test_split(
        range(len(train_data)),
        test_size=1-train_val_split,
        stratify=labels,
        random_state=random_seed
    )

    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    # 包装训练集和验证集
    train_subset = Subset(train_data, train_idx)
    val_subset = Subset(train_data, val_idx)
    train_data_split = TransformSubset(train_subset, transform=train_transform)
    val_data = TransformSubset(val_subset, transform=val_transform)

    # 训练
    train(device, train_data_split, val_dataset=val_data, test_dataset=test_data,
          num_epochs=num_epochs, batch_size=batch_size, lr=lr,
          model_path=model_path, use_wandb=use_wandb,
          patience=patience, min_delta=min_delta)