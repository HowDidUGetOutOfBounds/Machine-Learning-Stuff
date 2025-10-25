import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

## some experiments, see main 2.2 file for tasks

# === 1. Загрузка данных ===
data = pd.read_csv('ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'Passed'])
passed = data[data['Passed'] == 1]
failed = data[data['Passed'] == 0]

# === 2. Масштабирование признаков ===
X_raw = data[['Test 1', 'Test 2']].values
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X_scaled = (X_raw - X_mean) / X_std

y = data['Passed'].values


# === 3. Полиномиальные признаки ===
def map_feature(x1, x2, degree=6):
    x1, x2 = np.asarray(x1), np.asarray(x2)
    features = [np.ones(x1.shape[0])]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            features.append((x1 ** (i - j)) * (x2 ** j))
    return np.column_stack(features)


# === 4. Устойчивая сигмоида ===
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# === 5. L2-регуляризация ===
def cost_function_reg(theta, X, y, lambda_):
    m = y.size
    h = sigmoid(X.dot(theta))
    cost = -(1 / m) * (y.dot(np.log(h + 1e-8)) + (1 - y).dot(np.log(1 - h + 1e-8)))
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return cost + reg


def gradient_reg(theta, X, y, lambda_):
    m = y.size
    h = sigmoid(X.dot(theta))
    grad = (1 / m) * X.T.dot(h - y)
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad


def scipy_optimize(X, y, lambda_, method='BFGS'):
    theta_init = np.zeros(X.shape[1])
    res = optimize.minimize(
        fun=lambda t: cost_function_reg(t, X, y, lambda_),
        x0=theta_init,
        jac=lambda t: gradient_reg(t, X, y, lambda_),
        method=method,
        options={'maxiter': 4000, 'disp': False}
    )
    return res


# === 6. Функции предсказания ===
def predict_proba(theta, X):
    return sigmoid(X.dot(theta))


def predict_class(theta, X, threshold=0.5):
    return (predict_proba(theta, X) >= threshold).astype(int)


# === 7. График разделяющих кривых для разных λ ===
lambdas = [0, 1, 10, 100]
plt.figure(figsize=(16, 16))

for i, lambda_ in enumerate(lambdas, 1):
    # Полиномиальные признаки
    X_poly = map_feature(X_scaled[:, 0], X_scaled[:, 1], degree=6)
    theta_poly = scipy_optimize(X_poly, y, lambda_).x

    # Линейные признаки (degree=1)
    X_lin = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
    theta_lin = scipy_optimize(X_lin, y, lambda_).x

    # Сетка для графика
    u = np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 100)
    v = np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 100)
    U, V = np.meshgrid(u, v)

    # Полиномиальная вероятность
    Z_poly = sigmoid(np.dot(map_feature(U.ravel(), V.ravel(), degree=6), theta_poly)).reshape(U.shape)
    # Линейная вероятность
    Z_lin = sigmoid(theta_lin[0] + theta_lin[1] * U + theta_lin[2] * V)

    # Рисуем subplot
    plt.subplot(2, 2, i)
    plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], s=70, c='b', marker='+', label='Прошло')
    plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], s=70, c='r', marker='x', label='Не прошло')
    plt.contour(U, V, Z_poly, levels=[0.5], colors='g', linewidths=2, linestyles='-')
    plt.contour(U, V, Z_lin, levels=[0.5], colors='k', linewidths=2, linestyles='--')
    plt.title(f'λ={lambda_} | Полиномиальная: зелёная, Линейная: чёрная пунктир')
    plt.xlabel('Test 1 (масштаб)')
    plt.ylabel('Test 2 (масштаб)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()