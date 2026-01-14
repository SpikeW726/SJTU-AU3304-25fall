import numpy as np
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataset_stats(src_path, resize_shape=[224,224]):
    img_paths = [os.path.join(root, f) for root, _, files in os.walk(src_path) for f in files if f.endswith(('.jpg', '.png'))]
    print(f"  正在计算 {len(img_paths)} 张图片的均值和标准差...")

    # 流式计算均值和标准差,避免内存溢出
    # 先计算所有像素的累积和
    pixel_sum = 0.0
    pixel_sum_sq = 0.0
    total_pixels = 0

    for i, p in enumerate(img_paths):
        # 转灰度图并将H/W颠倒,因为PIL和torch的定义不同
        img = np.array(Image.open(p).convert('L').resize(resize_shape[::-1])) / 255.0

        # 累积像素值和平方值
        pixel_sum += img.sum()
        pixel_sum_sq += (img ** 2).sum()
        total_pixels += img.size

        if (i + 1) % 100 == 0:
            print(f"    已处理 {i + 1}/{len(img_paths)} 张图片...")

    # 计算全局均值和标准差 (标量)
    mean = pixel_sum / total_pixels
    variance = (pixel_sum_sq / total_pixels) - (mean ** 2)
    std = np.sqrt(variance) if variance > 0 else 0.0

    print(f"  均值: {mean:.4f}, 标准差: {std:.4f}")

    return float(mean), float(std)

def load_data(src:str, batch_size:int, resize_shape:list=[224,224], shuffle:bool=False):
    current_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_path, src)
    mean, std = get_dataset_stats(src_path, resize_shape)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])

    data = datasets.ImageFolder(src_path, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    class_names = data.classes
    print(class_names)
    class_idx = data.class_to_idx
    print(class_idx)

    return loader, class_names

def load_dataset(src: str, resize_shape: list = [224, 224]):
    current_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_path, src)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(src_path, transform=transform)
    return dataset

if __name__ == "__main__":
    loader, _ = load_data(src="rps", batch_size=32)
    data_iter = iter(loader)
    images, labels = next(data_iter)
    print(images.size())