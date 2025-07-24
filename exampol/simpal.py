import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Step 1: 创建简单的人工二分类数据集
np.random.seed(0)
X = np.random.randn(200, 2)  # 200个样本，每个样本有2个特征
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)  # 使用XOR逻辑生成标签

# 可视化数据集
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Generated Dataset')
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: 定义并编译模型
model = Sequential([
    Dense(4, input_dim=2, activation='relu'),  # 输入层 (2个特征) -> 隐藏层 (4个神经元)
    Dense(1, activation='sigmoid')  # 输出层 (1个神经元)，用于二分类
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: 训练模型
history = model.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test))

# Step 4: 评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'模型准确率: {accuracy * 100:.2f}%')

# 使用模型进行预测
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype("int32")

# 打印前几个预测结果对比
print("\n前几个预测结果对比:")
for i in range(5):
    print(
        f"预测: {'Class 1' if predicted_classes[i] == 1 else 'Class 0'}, 实际: {'Class 1' if y_test[i] == 1 else 'Class 0'}")


# Step 5: 可视化决策边界（可选）
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Decision Boundary')
    plt.show()


plot_decision_boundary(model, X, y)