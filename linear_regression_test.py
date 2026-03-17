import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[50],[80],[120],[150],[200]])
Y = np.array([100,160,240,300,380])
model = LinearRegression()
model.fit(X,Y)
N = float(input("Enter a value for N: "))
predicted = model.predict([[N]])
print("Predicted value for N =", predicted)

# 画散点图
plt.scatter(X, Y)

# 画回归直线
plt.plot(X, model.predict(X))

plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("House Price Prediction")

plt.show()