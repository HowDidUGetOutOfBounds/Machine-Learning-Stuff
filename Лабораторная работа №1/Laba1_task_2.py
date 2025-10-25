import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# 6. Загрузка данных
# =========================
data = np.loadtxt('ex1data2.txt', delimiter=',')

X = data[:, :2]  # признаки: площадь дома, количество комнат
y = data[:, 2]   # целевая переменная: стоимость дома
m = y.size

print("Первые 5 строк X:\n", X[:5])
print("Первые 5 значений y:\n", y[:5])

# =========================
# 7. Нормализация признаков
# =========================
def feature_normalize(X):
    """Нормализация признаков (среднее 0, стандартное отклонение 1)."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X_norm, mu, sigma = feature_normalize(X)
X_norm = np.concatenate([np.ones((m, 1)), X_norm], axis=1)  # добавляем столбец единиц для θ0

# =========================
# 8. Функции потерь и градиентного спуска (векторизация)
# =========================
def compute_cost(X, y, theta):
    """Векторизованная функция потерь."""
    m = y.size
    errors = X @ theta - y
    return (1 / (2 * m)) * np.dot(errors, errors)

def gradient_descent(X, y, theta, alpha, num_iters):
    """Векторизованный градиентный спуск."""
    m = y.size
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

# =========================
# Градиентный спуск (с нормализацией)
# =========================
theta = np.zeros(X_norm.shape[1])
alpha = 0.01
num_iters = 1000

theta, J_history = gradient_descent(X_norm, y, theta, alpha, num_iters)

plt.plot(range(num_iters), J_history, 'b-')
plt.xlabel('Итерации')
plt.ylabel('Функция потерь J(θ)')
plt.title('Сходимость градиентного спуска (с нормализацией)')
plt.show()

# =========================
# Сравнение сходимости с и без нормализации
# =========================
X_no_norm = np.concatenate([np.ones((m, 1)), X], axis=1)
theta_zero = np.zeros(X_no_norm.shape[1])

# Используем очень маленький alpha, чтобы избежать переполнения
theta_no_norm, J_no_norm = gradient_descent(X_no_norm, y, theta_zero, alpha=0.0000005, num_iters=num_iters)

plt.plot(range(num_iters), J_history, label='С нормализацией')
plt.plot(range(num_iters), J_no_norm, label='Без нормализации (α очень маленький)')
plt.xlabel('Итерации')
plt.ylabel('Функция потерь J(θ)')
plt.legend()
plt.title('Сравнение сходимости с/без нормализации')
plt.show()

# =========================
# 9. Прирост производительности от векторизации
# =========================
def gradient_descent_loops(X, y, theta, alpha, num_iters):
    """Градиентный спуск через циклы for (не векторизованный)."""
    m = y.size
    J_history = []
    for i in range(num_iters):
        predictions = np.zeros(m)
        for j in range(m):
            for k in range(theta.size):
                predictions[j] += theta[k] * X[j, k]
        errors = predictions - y
        temp = np.zeros(theta.size)
        for k in range(theta.size):
            for j in range(m):
                temp[k] += errors[j] * X[j, k]
            temp[k] = theta[k] - (alpha / m) * temp[k]
        theta = temp.copy()
        # Функция потерь
        J = sum((predictions - y)**2) / (2*m)
        J_history.append(J)
    return theta, J_history

theta_init = np.zeros(X_norm.shape[1])
num_iters_perf = 100
alpha_perf = 0.01

# Векторизованный
start_vec = time.time()
theta_vec, J_vec = gradient_descent(X_norm, y, theta_init.copy(), alpha_perf, num_iters_perf)
end_vec = time.time()

# Циклы
start_loop = time.time()
theta_loop, J_loop = gradient_descent_loops(X_norm, y, theta_init.copy(), alpha_perf, num_iters_perf)
end_loop = time.time()

print("Векторизация:", end_vec - start_vec, "сек")
print("Циклы for:", end_loop - start_loop, "сек")

plt.plot(range(num_iters_perf), J_vec, label='Векторизация')
plt.plot(range(num_iters_perf), J_loop, label='Циклы for')
plt.xlabel('Итерации')
plt.ylabel('J(θ)')
plt.legend()
plt.title('Прирост производительности: векторизация vs циклы')
plt.show()

# =========================
# 10. Влияние коэффициента обучения α
# =========================
alphas = [0.001, 0.01, 0.1, 0.3, 1]
plt.figure()
for a in alphas:
    theta_test = np.zeros(X_norm.shape[1])
    _, J_hist = gradient_descent(X_norm, y, theta_test, a, num_iters)
    plt.plot(range(num_iters), J_hist, label=f'alpha={a}')
plt.xlabel('Итерации')
plt.ylabel('J(θ)')
plt.legend()
plt.title('Влияние α на скорость сходимости')
plt.show()

# =========================
# 11. Аналитическое решение (Normal Equation)
# =========================
X_analytical = np.concatenate([np.ones((m, 1)), X], axis=1)
theta_analytical = np.linalg.pinv(X_analytical.T @ X_analytical) @ (X_analytical.T @ y)

print("θ (градиентный спуск, с нормализацией):", theta)
print("θ (аналитическое решение):", theta_analytical)

# Разнормализация θ после градиентного спуска
theta_orig = np.zeros_like(theta)
theta_orig[1:] = theta[1:] / sigma
theta_orig[0] = theta[0] - np.sum((theta[1:] * mu) / sigma)

print("θ (градиентный спуск, разнормализованные):", theta_orig)
print("θ (аналитическое решение):", theta_analytical)