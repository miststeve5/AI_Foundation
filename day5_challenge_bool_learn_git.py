import numpy as np

# 生成随机 RGB 图像 (100x100, 3)
img = np.random.randint(0, 256, (100, 100, 3))

# 逻辑挑战：提取 R > (G + B) 的像素点
# 提示：利用切片提取通道，再进行向量化比较
r_channel = img[:, :, 0]
g_channel = img[:, :, 1]
b_channel = img[:, :, 2]

# 这里的比较会触发广播机制
red_mask = r_channel > (g_channel + b_channel)

# 统计满足条件的像素占比
print(f"红光显著像素占比: {np.sum(red_mask) / (100*100):.2%}")