import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== 1) Загрузка данных ==========
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]   # население (10k)
y = data[:, 1]   # прибыль (10k)
m = len(y)

# Добавляем столбец единиц для θ0
X_b = np.c_[np.ones(m), X]

# ========== 2) Построение графика ==========
plt.figure(figsize=(8,6))
plt.scatter(X, y, c='red', marker='x')
plt.xlabel('Population (10,000s)')
plt.ylabel('Profit (10,000s)')
plt.title('Profit vs Population')
plt.grid(True)
plt.show()

# ========== 3) Функция потерь ==========
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    return (1/(2*m)) * np.sum(errors**2)

# Проверим начальную стоимость
theta_init = np.zeros(2)
print("Initial cost J([0,0]) =", compute_cost(X_b, y, theta_init))

# ========== 4) Градиентный спуск ==========
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        errors = X.dot(theta) - y
        theta -= (alpha/m) * X.T.dot(errors)
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

alpha = 0.01
num_iters = 1500
theta, J_history = gradient_descent(X_b, y, theta_init, alpha, num_iters)

print("Theta from gradient descent:", theta)

# Линия регрессии
plt.figure(figsize=(8,6))
plt.scatter(X, y, c='red', marker='x')
plt.plot(X, X_b.dot(theta), color='blue')
plt.xlabel('Population (10,000s)')
plt.ylabel('Profit (10,000s)')
plt.title('Linear regression fit')
plt.grid(True)
plt.show()

# График сходимости J(θ)
plt.figure(figsize=(8,6))
plt.plot(range(1, num_iters+1), J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J')
plt.title('Convergence of gradient descent')
plt.grid(True)
plt.show()

# ========== 5) Поверхность и контур J(θ0,θ1) ==========
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        J_vals[i, j] = compute_cost(X_b, y, np.array([t0, t1]))

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

# Поверхность
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals.T, cmap='viridis')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('Cost J')
plt.title('Cost function surface')
plt.show()

# Контур
plt.figure(figsize=(8,6))
plt.contour(T0, T1, J_vals.T, levels=np.logspace(-2, 3, 20))
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)  # найденные θ
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Cost function contour')
plt.show()