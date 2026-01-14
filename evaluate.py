import torch
from torch import nn
from network import simpleVGG
from test import evaluate
from data_loader import load_dataset

def load_and_evaluate(model_path, dataset, batch_size, device, return_loss=False, loss_fn=None, verbose=True):
    """
    从指定路径加载模型并在数据集上评估
    如果return_loss=False: 返回accuracy
    如果return_loss=True: 返回 (accuracy, loss)
    """
    # 创建模型实例
    model = simpleVGG().to(device)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)

    # 判断是完整checkpoint还是只有state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整checkpoint
        state_dict = checkpoint['model_state_dict']
        if verbose:
            epoch = checkpoint.get('epoch', 'unknown')
            val_acc = checkpoint.get('val_acc', 'unknown')
            print(f"Checkpoint from epoch {epoch}, val_acc in valuate dataset: {val_acc:.2f}%" if val_acc != 'unknown' else f"Checkpoint from epoch {epoch}")
    else:
        # 只有state_dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    if verbose:
        print(f"Model loaded from: {model_path}")

    # 调用 evaluate 函数
    return evaluate(model, dataset, batch_size, device, return_loss=return_loss,
                   loss_fn=loss_fn, verbose=verbose)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    test_data = load_dataset("rps-test-set")
    loss_fn = nn.CrossEntropyLoss()

    # 加载模型并在测试集上评估
    test_acc, test_loss = load_and_evaluate(
        model_path=r"results\models5\train_model_20260114_020449.pth",
        dataset=test_data,
        batch_size=32,
        device=device,
        return_loss=True,
        loss_fn=loss_fn,
        verbose=True
    )
    print(f"Final Test Accuracy: {test_acc:.2f}%")