import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

current_path = "C:\\Users\\Leyan\\OneDrive\\Traffic Sign Dataset"

# 假设数据集的根目录为 'dataset'，包含多个类别的图像
dataset_path = current_path + '/traffic_Data/DATA/'

# 使用 ImageFolder 加载数据集
dataset = ImageFolder(dataset_path, transform=transforms.ToTensor())

# 初始化每个通道的累加值和像素数量
sum_channel_values = torch.zeros(3)
sum_channel_squared_values = torch.zeros(3)
total_pixels = 0

df = pd.read_csv(os.path.join(current_path, "distinct_labels.csv"))

# # 遍历数据集
# for image, _ in tqdm(dataset):
#     total_pixels += image.size(1) * image.size(2)  # 像素数量累加
#     sum_channel_values += torch.sum(image, dim=(1, 2))  # 对每个通道进行累加
#     sum_channel_squared_values += torch.sum(image ** 2, dim=(1, 2))  # 对每个通道的平方进行累加

# # 计算每个通道的平均值
# mean_values = sum_channel_values / total_pixels

# # 计算每个通道的方差
# variance_values = (sum_channel_squared_values / total_pixels) - (mean_values ** 2)

# # 计算每个通道的标准差
# stddev_values = torch.sqrt(variance_values)

# # 打印结果
# print("每个通道的平均值:", mean_values)
# print("每个通道的标准差:", stddev_values)

# # 每个通道的平均值: tensor([0.4246, 0.4163, 0.4216])
# # 每个通道的标准差: tensor([0.2405, 0.2302, 0.2444])


from PIL import Image
import os
from tqdm import tqdm
from glob import glob

# 图像文件夹路径
image_folder = current_path + '/traffic_Data/DATA/*/*'

# 初始化总宽度和总高度
total_width = 0
total_height = 0

# 初始化图像计数器
image_count = 0
labels = {i:0 for i in range(58)}

# 遍历图像文件夹
for filename in tqdm(glob(image_folder)):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        # labels.append(int(image_path.split("\\")[-2]))
        
        labels[int(image_path.split("\\")[-2])] = labels[int(image_path.split("\\")[-2])] + 1
        # 打开图像
        with Image.open(image_path) as img:
            # 累加宽度和高度
            total_width += img.width
            total_height += img.height
            # 更新图像计数
            image_count += 1

# 计算平均宽度和平均高度
average_width = total_width / image_count
average_height = total_height / image_count

print(f"Average Width: {average_width}, Average Height: {average_height}")
# Average Width: 152.13717026378896, Average Height: 140.85659472422063


# 初始化每个通道的累加值和像素数量
sum_channel_values = torch.zeros(3)
sum_channel_squared_values = torch.zeros(3)
total_pixels = 0

# 遍历数据集
for image, _ in tqdm(dataset):
    total_pixels += image.size(1) * image.size(2)  # 像素数量累加
    sum_channel_values += torch.sum(image, dim=(1, 2))  # 对每个通道进行累加
    sum_channel_squared_values += torch.sum(image ** 2, dim=(1, 2))  # 对每个通道的平方进行累加

# 计算每个通道的平均值
mean_values = sum_channel_values / total_pixels

# 计算每个通道的方差
variance_values = (sum_channel_squared_values / total_pixels) - (mean_values ** 2)

# 计算每个通道的标准差
stddev_values = torch.sqrt(variance_values)

# 打印结果
print("每个通道的平均值:", mean_values)
print("每个通道的标准差:", stddev_values)

# 每个通道的平均值: tensor([0.4246, 0.4163, 0.4216])
# 每个通道的标准差: tensor([0.2405, 0.2302, 0.2444])

def calculate_focal_loss_weights(labels, alpha=0.25, gamma=2.0):
    # 計算每個類別的權重
    class_counts = np.bincount(labels)
    total_samples = len(dataset)
    class_weights = total_samples / (len(class_counts) * class_counts.astype(float))

    # 正規化權重
    class_weights /= np.sum(class_weights)

    # 計算每個樣本的權重
    weights = np.zeros_like(labels, dtype=float)
    for i, label in enumerate(labels):
        weights[i] = class_weights[label]

    # 計算 Focal Loss 的最終權重
    focal_weights = alpha * np.power(1 - class_weights, gamma)

    return weights * focal_weights

# labels = np.array(list(labels.values()))

class_samples = list(labels.values())  # 替換為實際類別樣本數
total_samples = sum(class_samples)
weights = [total_samples / (len(class_samples) * c) for c in class_samples]
weight_tensor = torch.FloatTensor(weights)



print("標籤:", labels)
print("權重:", weights)   

# 標籤: {0: 118, 1: 40, 2: 80, 3: 260, 4: 98, 5: 194, 6: 78, 7: 152, 8: 8, 9: 2, 10: 70, 11: 138, 12: 96, 13: 36, 14: 128, 15: 22, 16: 142, 17: 130, 18: 8, 19: 4, 20: 18, 21: 12, 22: 18, 23: 14, 24: 100, 25: 2, 26: 126, 
# 27: 28, 28: 446, 29: 44, 30: 150, 31: 42, 32: 14, 33: 4, 34: 26, 35: 156, 36: 40, 37: 58, 38: 30, 39: 34, 40: 32, 41: 18, 42: 32, 43: 82, 44: 30, 45: 24, 46: 18, 47: 12, 48: 10, 49: 42, 50: 56, 51: 8, 52: 36, 53: 2, 54: 324, 55: 162, 56: 110, 57: 6}

# 權重: [0.6092928112215079, 1.7974137931034482, 0.8987068965517241, 0.276525198938992, 0.7336382828993666, 0.3706007820831852, 
# 0.9217506631299734, 0.47300362976406535, 8.987068965517242, 35.94827586206897, 1.0270935960591132, 0.5209895052473763, 
# 0.7489224137931034, 1.9971264367816093, 0.5616918103448276, 3.268025078369906, 0.5063137445361826, 0.553050397877984, 
# 8.987068965517242, 17.974137931034484, 3.9942528735632186, 5.991379310344827, 3.9942528735632186, 5.135467980295567, 
# 0.7189655172413794, 35.94827586206897, 0.5706075533661741, 2.5677339901477834, 0.16120303077160972, 1.634012539184953, 
# 0.4793103448275862, 1.7118226600985222, 5.135467980295567, 17.974137931034484, 2.7652519893899203, 0.4608753315649867, 
# 1.7974137931034482, 1.2395957193816884, 2.396551724137931, 2.114604462474645, 2.2467672413793105, 3.9942528735632186, 
# 2.2467672413793105, 0.8767872161480236, 2.396551724137931, 2.9956896551724137, 3.9942528735632186, 5.991379310344827, 
# 7.189655172413793, 1.7118226600985222, 1.2838669950738917, 8.987068965517242, 1.9971264367816093, 35.94827586206897, 
# 0.2219029374201788, 0.4438058748403576, 0.6536050156739812, 11.982758620689655]