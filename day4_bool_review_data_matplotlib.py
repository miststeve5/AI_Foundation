import numpy as np
import matplotlib.pyplot as plt

# 显示时长（秒）：如果看不清，可以调大
SHOW_SECONDS = 4

def show_pause_close(seconds: int = SHOW_SECONDS) -> None:
    plt.show(block=False)
    plt.pause(seconds)
    plt.close()

# 说明：本文件以“笔记式注释”为主，帮助你复盘思路与机制
# ==============================================================================
# 复盘大纲
# 1) 机器人视觉：暗部涂黑 (布尔索引 + 广播)
# 2) 随机数单板块：统计验证
# 3) 布尔索引单板块：颜色筛选 + 离群清洗
# 4) 堆叠/拆分：2D + 3D 形状变化
# 5) Matplotlib：子图、直方图、图像显示
# 6) 流程 A/B：两个完整流程对比
# ==============================================================================

# ==============================================================================
# === 复盘：机器人视觉系统 ===
# 目标：自动识别并“涂黑”所有过暗（亮度低于 100）的区域
# ==============================================================================
# 1. 模拟一个彩色图像 (4x4, 3 通道)
# 数值范围 0-255，对应像素亮度
np.random.seed(0)
img = np.random.randint(0, 256, size=(4, 4, 3))
#这里的 img 是一个 4x4 的小图，每个像素有 R/G/B 三个通道，数值随机在 0-255 之间。
# 这个时候 img 的形状是 (4, 4, 3)，表示有 4 行 4 列，每个像素有 3 个颜色通道。

print("--- 原始 RGB 像素 (取左上角第一个点) ---")
print(img[0, 0]) # 例如 [172, 47, 117]

# 2. 挖掘深度：计算“亮度” (对 axis=2 求均值)
# axis=2 是通道维度 (R,G,B)
# keepdims=True 让输出形状仍是 (H, W, 1)，便于和原图广播
brightness = img.mean(axis=2, keepdims=True)
print(f"\n亮度矩阵形状: {brightness.shape}") # (4, 4, 1)

# 3. 布尔索引实战：找出亮度低于 100 的像素
# dark_mask 形状 (4, 4, 1)，每个位置 True/False
# squeeze() 把 (4,4,1) 压成 (4,4)，方便做布尔索引

dark_mask = brightness < 100
# 这里的 dark_mask 是一个布尔矩阵，True 的位置表示对应像素的亮度低于 100，也就是过暗的区域。
# 这个时候 dark_mask 的形状是 (4, 4, 1)，因为我们在计算亮度的时候设置了 keepdims=True。
# 如果我们不需要保留这个维度，可以使用 squeeze() 方法把它压缩掉，得到一个形状为 (4, 4) 的布尔矩阵。
# 等价写法：dark_mask[:, :, 0] 也能得到 (4, 4) 的掩码。

# 4. 向量化修改：把暗部区域全部变成纯黑色 [0, 0, 0]
# 这里的广播机制会让 [0, 0, 0] 填满所有满足条件的通道位置
img[dark_mask.squeeze()] = 0
# 这里的 img[dark_mask.squeeze()] 是一个布尔索引操作
# 它会选出所有 dark_mask 中为 True 的位置对应的像素点。
# 由于 dark_mask 的形状是 (4, 4)，它会选出所有满足条件的像素点的行列位置
# 然后把这些位置对应的 img 中的像素值设置为 [0, 0, 0]，也就是纯黑色。
# 这个时候，原来亮度低于 100 的像素点现在已经被修改成了黑色了。

print("\n处理后的图像 (暗部已清零):")
print(img)

# ==============================================================================
# 随机数单板块深挖
# ==============================================================================
# 设定种子，确保结果可复现
np.random.seed(42)

# 模拟具身智能三轴加速度计数据 (Time_Steps=1000, Channels=3)
# randn = 标准正态分布 N(0,1)
# 1000 表示 1000 个时刻，3 表示 x/y/z 三轴
sensor_data = np.random.randn(1000, 3)

# --- 深度验证 ---
# axis=0 压缩时间轴 -> 得到每个通道的统计量
means = sensor_data.mean(axis=0)
stds = sensor_data.std(axis=0)
# 这里的 means 是一个长度为 3 的数组，分别表示 x/y/z 三轴的平均值。
# 由于我们使用的是标准正态分布生成的数据，所以理论上每个通道的均值应该接近 0，标准差应该接近 1。
# 这个时候 stds 也是一个长度为 3 的数组，分别表示 x/y/z 三轴的标准差。
# 通过这个统计验证，我们可以确认我们生成的随机数据确实符合我们预期的分布特性，这对于后续的分析和建模是非常重要的。


print("--- 阶段 1：随机数据质量验证 ---")
print(f"三轴均值 (应接近0): {means}")
print(f"三轴标准差 (应接近1): {stds}")

# ==============================================================================
# 布尔索引单板块深挖 1：颜色筛选
# ==============================================================================
# 模拟一个 100x100 的随机 RGB 图片
img = np.random.randint(0, 256, size=(100, 100, 3))

# 挖深：提取所有“偏红”的像素
# 条件：R > 200 且 G < 100 且 B < 100
# 注意：每个通道都是 (100, 100) 的矩阵，& 会逐元素计算
# red_mask 形状是 (100, 100)，True 表示“偏红”
red_mask = (img[:, :, 0] > 200) & (img[:, :, 1] < 100) & (img[:, :, 2] < 100)
# red_mask 是一个布尔矩阵，每个位置的值是 True 或 False。
# True 的位置表示对应像素满足“偏红”的条件，也就是 R 通道的值大于 200，G 和 B 通道的值都小于 100。
# img里的冒号表示我们要对所有的行和列进行操作，而最后的数字 0、1、2 分别表示 R、G、B 三个通道。

# 统计比例：True 在 NumPy 中可视为 1
red_pixel_count = np.sum(red_mask)
# 这里的 np.sum(red_mask) 会把 red_mask 中的 True 视为 1，False 视为 0
# 所以它会计算出满足条件的像素点的总数，也就是“偏红”像素的数量。
total_pixels = img.shape[0] * img.shape[1]
# 这里的 total_pixels 是图像中总的像素数量，等于行数乘以列数，也就是 100 * 100 = 10000。
print(f"\n--- 阶段 2：3D 布尔过滤 ---")
print(f"红色像素数量: {red_pixel_count} | 占比: {red_pixel_count/total_pixels:.2%}")

# 应用掩码：创造一个只有红色的提取图
# 先建立黑底，再把满足条件的像素“点名”复制过去
red_only = np.zeros_like(img)
# 这里的 red_only 是一个全黑的图像，形状和 img 一样，也是 (100, 100, 3)
# 但是所有像素值都是 [0, 0, 0]。
red_only[red_mask] = img[red_mask] # 这种“名单点名法”不会产生循环

# ==============================================================================
# 布尔索引单板块深挖 2：离群点清洗
# ==============================================================================
# 模拟：原始信号 (均值 50) + 极端离群噪声
raw_signal = np.random.randn(500) * 10 + 50
# 人为制造一些物理上不可能的“坏点”（比如传感器瞬间断电返回 -999）
raw_signal[np.random.randint(0, 500, 10)] = -999

# 1. 生成清洗掩码
valid_mask = (raw_signal > 0) & (raw_signal < 150) 
# 这里的 valid_mask 是一个布尔数组
# True 的位置表示对应的 raw_signal 中的值在 0 到 150 之间，也就是我们认为的“有效数据”。
# 这个时候 valid_mask 的形状是 (500,)
# 每个位置的值是 True 或 False，表示对应的 raw_signal 中的值是否有效。

# 2. 统计剔除比例 (工程必备)
# ~ 取反 -> 选出无效点
removed_count = np.sum(~valid_mask) 
# 这里的 ~valid_mask 是 valid_mask 的取反，也就是把 True 变成 False，False 变成 True。
loss_rate = removed_count / len(raw_signal)
# 这里的 removed_count 是无效点的数量，也就是被剔除的离群点的数量。
# len(raw_signal) 是原始信号的总长度，也就是 500。
# loss_rate 是剔除的离群点占总数据的比例
# 这个指标对于评估数据质量和清洗效果非常重要，过高的剔除率可能意味着数据质量有问题或者清洗条件过于严格。

# 3. 提取干净数据
# clean_signal 是“清洗后可用的数据”，可用于后续统计/建模
clean_signal = raw_signal[valid_mask] 

print(f"\n--- 阶段 3：离群点清洗统计 ---")
print(f"剔除异常点个数: {removed_count}")
print(f"数据损耗率: {loss_rate:.2%}")
print(f"清洗后信号均值: {clean_signal.mean():.2f}")

# ==============================================================================
# 新知识：堆叠 (Stacking)
# ==============================================================================
# 准备两个 2x2 的“方块”
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 垂直拼 (Vertical) -> 变成 4x2
# 维度变化：行数叠加
v_stack = np.vstack((a, b))
#变成了 4 行 2 列，第一行到第二行是 a 的内容，第三行到第四行是 b 的内容。
#也就是 [[1, 2],
#       [3, 4],
#       [5, 6],
#       [7, 8]]

# 水平拼 (Horizontal) -> 变成 2x4
# 维度变化：列数叠加
h_stack = np.hstack((a, b))
#变成了 2 行 4 列，第一列到第二列是 a 的内容，第三列到第四列是 b 的内容。
#也就是 [[1, 2, 5, 6],
#       [3, 4, 7, 8]]

# 万能拼 (axis=0 是垂直, axis=1 是水平)
c_stack = np.concatenate((a, b), axis=1) # 效果等同于 hstack

# ==============================================================================
# 新知识：拆分 (Splitting)
# ==============================================================================
# 把刚才 4x2 的 v_stack 均分成两份
upper, lower = np.vsplit(v_stack, 2)
#变成了两行，每行是一个 2x2 的方块，第一行是 a，第二行是 b。
#也就是 upper = [[1, 2],
#              [3, 4]]
#    lower = [[5, 6],
#             [7, 8]]

# 把刚才 2x4 的 h_stack 均分成两份
left, right = np.hsplit(h_stack, 2)
#变成了两列，每列是一个 2x2 的方块，第一列是 a，第二列是 b。
#也就是 left = [[1, 2],
#              [3, 4]]
#    right = [[5, 6],
#             [7, 8]]

# ==============================================================================
# 三维的堆叠与拆分
# ==============================================================================

# 1. 创建两个模拟的 3D 数据块 (2x2x3, 比如两个小像素块)
block1 = np.ones((2, 2, 3)) * 1  # 全是 1 的块
block2 = np.ones((2, 2, 3)) * 2  # 全是 2 的块

print("--- 3D 堆叠实验 ---")
# 水平拼 (拼成左右眼)
# 这里的 hstack 会在“宽度”方向扩展
h_combined = np.hstack((block1, block2))
# 结果是一个 2x4x3 的块：每一行都变成 [block1 | block2] 的左右拼接
# 也就是每行从 2 列扩展到 4 列，但通道数仍是 3
# 这里的 hstack 在第三维（通道维度）上没有变化，只在“宽度”方向扩展
print(f"水平拼接形状: {h_combined.shape}") # (2, 4, 3)

# 深度拼 (增加特征通道)
# 注意：hstack/vstack 只能拼前两个轴，第三个轴必须用 concatenate
# axis=2 表示在“通道维度”上堆叠

depth_combined = np.concatenate((block1, block2), axis=2)
# 结果是一个 2x2x6 的块：每个像素的通道从 [1,1,1] 变成 [1,1,1,2,2,2]
#也就是 [[[1, 1, 1, 2, 2, 2],
#       [1, 1, 1, 2, 2, 2]],
#      [[1, 1, 1, 2, 2, 2],
#       [1, 1, 1, 2, 2, 2]]]
#这里的 concatenate 在第三维（通道维度）上扩展了，从 3 扩展到了 6
# 而在前两维（行数和列数）上没有变化，仍然是 2x2。 
print(f"深度拼接形状: {depth_combined.shape}") # (2, 2, 6)

# 2. 3D 拆分实验
# 把刚才 2x4x3 的画面从中切开
left_part, right_part = np.hsplit(h_combined, 2)
print(f"\n--- 3D 拆分实验 ---")
print(f"拆分后左侧形状: {left_part.shape}") # (2, 2, 3)

# ==============================================================================
# 新知识：Matplotlib
# ==============================================================================
# 创建一个宽 10 高 4 的画板
plt.figure(figsize=(10, 4))

# 绘制第 1 个子图 (1行2列中的第1个)
plt.subplot(1, 2, 1)
x = np.linspace(0, 10, 50) # 生成 0-10 之间的 50 个点
plt.plot(x, np.sin(x), color='blue', label='Sine Wave')
plt.title("Signal Waveform")
plt.legend()

# 绘制第 2 个子图 (1行2列中的第2个)
plt.subplot(1, 2, 2)
data = np.random.randn(1000)
plt.hist(data, bins=30, color='green', alpha=0.6) #bins 是柱子数量，alpha 是透明度
plt.title("Noise Distribution")

plt.tight_layout() # 自动调整布局，防止标题重叠
show_pause_close()

# 假设 img 是我们之前提到的 (100, 100, 3) 数组
img = np.random.randint(0, 256, size=(20, 20, 3)) 

plt.imshow(img)
plt.axis('off') # 关掉坐标轴，让它看起来更像照片
show_pause_close()

# ==============================================================================
# 多图对比
# ==============================================================================
# 准备数据
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

# 核心：plt.subplot(行, 列, 编号)
plt.figure(figsize=(10, 4))

# 左侧图：信号波形
plt.subplot(1, 2, 1)
plt.plot(x, y1, 'r--', label='Sine') # 'r--' 是红色虚线，label 是图例标签
plt.plot(x, y2, 'b-', label='Cosine')
plt.title("Waveform Contrast")
plt.legend()

# 右侧图：随机噪声分布 (验证你的随机数理解)
plt.subplot(1, 2, 2)
noise = np.random.randn(1000)
plt.hist(noise, bins=30, color='g', alpha=0.6)
plt.title("Normal Distribution Noise")

plt.tight_layout() # 自动调整间距，防止文字重叠
show_pause_close()

# ==============================================================================
# 流程 A：传感器异常清洗 -> 数据拼接 -> 可视化监控
# ==============================================================================
# 目标：完整演示“生成数据 -> 清洗 -> 拼接 -> 可视化”的工程流程
# 注意：这是 1D 传感器数据流程，不是图像流程

# 设置随机种子，保证实验复现
np.random.seed(42)

# 1) 生成传感器数据 + 注入离群点
flowA_raw = np.random.randn(100) * 10 + 50  # 均值 50，标准差 10
flowA_raw[np.random.randint(0, 100, 5)] = -999  # 注入异常点

# 2) 布尔索引清洗
flowA_mask = flowA_raw > 0
flowA_filtered = flowA_raw[flowA_mask]
# 注意：这里只做“下限”清洗（>0），如果担心过大离群值，可再加上限条件
print("\n--- 流程 A-1: 传感器清洗 ---")
print(f"清洗前: {len(flowA_raw)} | 清洗后: {len(flowA_filtered)} | 剔除: {len(flowA_raw)-len(flowA_filtered)}")

# 3) 数据拼接：模拟双目相机左右画面
# 用极简 2x2 方块演示拼接规则（便于直接看懂形状变化）
# 这里的拼接是“形状演示”，不依赖上一步的 1D 传感器数据
flowA_left = np.array([[1, 2], [3, 4]])
flowA_right = np.array([[5, 6], [7, 8]])
flowA_stereo = np.hstack((flowA_left, flowA_right))
print("--- 流程 A-2: 数据拼接 ---")
print(f"左画面形状: {flowA_left.shape} | 右画面形状: {flowA_right.shape} | 拼接后形状: {flowA_stereo.shape}")

# 4) 可视化：清洗后的传感器分布 + 线性关系示意
# 左图看“分布”，右图看“趋势”
print("--- 流程 A-3: 可视化 ---")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(flowA_filtered, bins=20, color='teal', edgecolor='black', alpha=0.7)
plt.title("Flow A: Cleaned Sensor Distribution")
plt.xlabel("Distance")
plt.ylabel("Frequency")

x = np.linspace(0, 10, 50)
y = 2.5 * x + 1.2 + np.random.randn(50) * 2
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='tomato', s=20, label='Noisy Data')
plt.plot(x, 2.5 * x + 1.2, color='darkblue', linewidth=2, label='True Line')
plt.title("Flow A: Linear Trend Preview")
plt.legend()

plt.tight_layout()
show_pause_close()

# ==============================================================================
# 流程 B：机器人视觉预处理模拟
# ==============================================================================
# 这是“图像流程”，与流程 A 的数据类型不同

# 设置种子，确保科研可复现
np.random.seed(42)

# ==============================================================================
# 第一步：多维随机数据生成与“体检” (随机数挖深)
# ==============================================================================
# 模拟：生成两个摄像头的原始随机噪声图像 (100x100, 3通道)
raw_cam_l = np.random.randint(0, 256, size=(100, 100, 3))
raw_cam_r = np.random.randint(0, 256, size=(100, 100, 3))

# 统计验证：如果均值严重偏离 127.5，说明模拟数据不均匀
print(f"--- 阶段 1: 数据体检 ---")
print(f"左相机 R通道均值: {raw_cam_l[:,:,0].mean():.2f}")
# 备注：randint(0,256) 理论均值约 127.5，样本越大越接近

# ==============================================================================
# 第二步：多路信号融合 (堆叠新知)
# ==============================================================================
# 模拟：将左右相机画面水平拼接，形成“双目视觉”
stereo_view = np.hstack((raw_cam_l, raw_cam_r))
print(f"--- 阶段 2: 画面融合 ---")
print(f"双目画面形状: {stereo_view.shape}") # 应为 (100, 200, 3)
# 这里的拼接只发生在宽度方向，通道数不变

# ==============================================================================
# 第三步：3D 空间特征提取 (布尔索引挖深)
# ==============================================================================
# 目标：在融合后的画面中，只提取出“高亮度”区域 (R,G,B 均大于 180)
# 这模拟了机器人寻找光源或高亮反光物体的过程
# bright_mask 是二维掩码，True 的位置代表“高亮像素”
bright_mask = (stereo_view[:,:,0] > 180) & (stereo_view[:,:,1] > 180) & (stereo_view[:,:,2] > 180)
# 注意：mask 只有 (H, W)，因为我们用通道条件压缩成“像素级”判断
# 阈值 180 是人为设定，越高越严格，命中率通常会很低

# 统计剔除比例
hit_rate = np.sum(bright_mask) / (stereo_view.shape[0] * stereo_view.shape[1])
# 这里的分母是 H*W（像素数），不是 H*W*3（通道数）
print(f"--- 阶段 3: 特征提取 ---")
print(f"高亮目标像素占比: {hit_rate:.2%}")

# 应用掩码：背景涂黑，只保留目标
# 先全黑，再把 True 位置的像素复制回来
extracted_targets = np.zeros_like(stereo_view)
extracted_targets[bright_mask] = stereo_view[bright_mask]

# ==============================================================================
# 第四步：全流程可视化监控 (Matplotlib 新知)
# ==============================================================================
plt.figure(figsize=(15, 6))

# 子图 1：展示原始双目拼接画面
plt.subplot(2, 1, 1)
plt.imshow(stereo_view)
plt.title("Step 1 & 2: Raw Stereo Concatenation (HStack)")
plt.axis('off')

# 子图 2：展示布尔过滤后的特征提取图
plt.subplot(2, 2, 3)
plt.imshow(extracted_targets)
plt.title("Step 3: Feature Extraction (Boolean Masking)")
plt.axis('off')

# 子图 3：展示亮度分布直方图
# flatten() 把 3D 图像拉直成 1D，便于统计像素强度分布
# 这里的直方图是“所有通道混在一起”的强度分布，不是灰度图
plt.subplot(2, 2, 4)
plt.hist(stereo_view.flatten(), bins=50, color='gray', alpha=0.7)
plt.axvline(180, color='red', linestyle='--', label='Threshold')
plt.title("Step 4: Pixel Intensity Distribution")
plt.legend()

plt.tight_layout()
show_pause_close()

# ==============================================================================
# 💡 核心认知小结
# 1) 布尔索引 = “名单点名”，不会写 for 循环
# 2) axis 维度压缩 = “压扁某个方向”，看清剩下什么
# 3) 堆叠/拆分 = 形状在变，通道/空间/时间要分清
# 4) 流程 A 是 1D 传感器，流程 B 是 3D 图像
# ==============================================================================
