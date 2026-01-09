import numpy as np
import cv2
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataset_stats(src_path, resize_shape=[224,224]):
    img_paths = [os.path.join(root, f) for root, _, files in os.walk(src_path) for f in files if f.endswith(('.jpg', '.png'))]
    
    imgs = []
    for p in img_paths:
        # 转灰度图并将H/W颠倒,因为PIL和torch的定义不同
        img = Image.open(p).convert('L').resize(resize_shape[::-1])
        imgs.append(np.array(img))
    
    imgs_array = np.stack(imgs) / 255.0
    mean = imgs_array.mean()
    std = imgs_array.std()
    
    return mean, std

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
    mean, std = get_dataset_stats(src_path, resize_shape)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])

    dataset = datasets.ImageFolder(src_path, transform=transform)
    return dataset

if __name__ == "__main__":
    loader, _ = load_data(src="rps", batch_size=32)
    data_iter = iter(loader)
    images, labels = next(data_iter)
    print(images.size())