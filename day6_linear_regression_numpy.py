import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# NumPy 手写线性回归（梯度下降法）
# 目标：不用 sklearn，纯用 NumPy 实现 y = wx + b 的拟合
# =============================================================

# ===== 第一步：生成模拟数据 =====
np.random.seed(42)  # 固定随机种子，保证每次运行结果一致

# 真实参数：y = 3x + 7 + 噪声
x = np.linspace(0, 10, 50)          # 50个均匀分布在0-10的点
y_true = 3 * x + 7                  # 真实的线性关系
noise = np.random.randn(50) * 2     # 标准正态噪声，乘以2控制噪声大小
y = y_true + noise                  # 加上噪声后的观测值

print("数据生成完毕")
print(f"x 的范围: {x.min():.1f} ~ {x.max():.1f}")
print(f"y 的范围: {y.min():.1f} ~ {y.max():.1f}")


# ===== 第二步：初始化参数 =====
w = 0.0   # 斜率，初始为0
b = 0.0   # 截距，初始为0
lr = 0.01 # 学习率（learning rate）：每次更新参数的步长
epochs = 1000  # 迭代次数：重复"预测→算损失→算梯度→更新"的循环次数

# 用一个列表记录每次迭代的损失值，后面画loss曲线用
loss_history = []


# ===== 第三步：梯度下降训练 =====
for epoch in range(epochs):
    # 1. 前向传播：用当前的 w 和 b 计算预测值
    y_pred = w * x + b
    
    # 2. 计算损失：MSE（均方误差）= mean((预测值 - 真实值)²)
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)
    
    # 3. 计算梯度（MSE对w和b的偏导数）
    # dL/dw = mean(2 * (y_pred - y) * x)
    # dL/db = mean(2 * (y_pred - y))
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))
    
    # 4. 更新参数：沿梯度的反方向走一小步
    w = w - lr * dw
    b = b - lr * db
    
    # 每100次打印一次进度
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

# 训练结束，打印最终结果
print(f"\n训练完成！")
print(f"学到的参数: w = {w:.4f}, b = {b:.4f}")
print(f"真实参数:   w = 3.0000, b = 7.0000")
print(f"最终损失:   {loss_history[-1]:.4f}")


# ===== 第四步：可视化 =====
plt.figure(figsize=(12, 5))

# 左图：数据散点图 + 拟合直线
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='steelblue', s=30, alpha=0.7, label='观测数据')
plt.plot(x, w * x + b, color='red', linewidth=2, label=f'拟合: y={w:.2f}x+{b:.2f}')
plt.plot(x, y_true, color='green', linewidth=1, linestyle='--', label='真实: y=3x+7')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()

# 右图：损失曲线（loss随迭代次数的变化）
plt.subplot(1, 2, 2)
plt.plot(loss_history, color='tomato', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
# 标注最终loss值
plt.annotate(f'Final: {loss_history[-1]:.2f}', 
             xy=(len(loss_history)-1, loss_history[-1]),
             fontsize=10, color='tomato')

plt.tight_layout()
plt.show()
