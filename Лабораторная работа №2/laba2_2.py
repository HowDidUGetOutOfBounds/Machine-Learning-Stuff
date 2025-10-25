import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

# === 1. Загрузка данных ===
data = pd.read_csv('ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'Passed'])

# === 2. Разделение по классам ===
passed = data[data['Passed'] == 1]
failed = data[data['Passed'] == 0]

# === 3. Визуализация === (Пункт 8)
plt.figure(figsize=(8, 6))
plt.scatter(passed['Test 1'], passed['Test 2'],
            s=70, c='b', marker='+', label='Контроль пройден')
plt.scatter(failed['Test 1'], failed['Test 2'],
            s=70, c='r', marker='x', label='Контроль не пройден')

plt.xlabel('Результат теста 1')
plt.ylabel('Результат теста 2')
plt.legend()
plt.title('Результаты тестов и прохождение контроля')
plt.grid(True)
plt.show()


# === 4. Создание полиномиальных признаков (Пункт 9) ===
def map_feature(x1, x2, degree=6):
    """
    Создаёт все полиномиальные комбинации признаков x1 и x2 до указанной степени.
    Возвращает DataFrame с 28 признаками при degree=6.
    """
    x1, x2 = np.asarray(x1), np.asarray(x2)
    features = [np.ones(x1.shape[0])]
    col_names = ['1']
    for i in range(1, degree + 1):
        for j in range(i + 1):
            features.append((x1 ** (i - j)) * (x2 ** j))
            col_names.append(f'x1^{i - j}x2^{j}')
    return pd.DataFrame(np.column_stack(features), columns=col_names)


X_mapped = map_feature(data['Test 1'], data['Test 2'], degree=6)
print("Размер расширенного набора признаков:", X_mapped.shape)
print("Пример первых столбцов:\n", X_mapped.iloc[:5, :7])


# === 5. Реализация логистической регрессии с L2-регуляризацией (Пункт 10) ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function_reg(theta, X, y, lambda_):
    m = y.size
    h = sigmoid(X.dot(theta))
    cost = -(1 / m) * (y.dot(np.log(h + 1e-8)) + (1 - y).dot(np.log(1 - h + 1e-8)))
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))  # регуляризация, кроме θ0
    return cost + reg


def gradient_reg(theta, X, y, lambda_):
    m = y.size
    h = sigmoid(X.dot(theta))
    grad = (1 / m) * X.T.dot(h - y)
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]
    return grad


def gradient_descent_reg(X, y, lambda_=1.0, alpha=0.1, num_iters=5000, tol=1e-6, verbose=True):
    theta = np.zeros(X.shape[1])
    J_history = []

    for i in range(num_iters):
        grad = gradient_reg(theta, X, y, lambda_)
        theta -= alpha * grad

        J = cost_function_reg(theta, X, y, lambda_)
        J_history.append(J)

        if verbose and i % (num_iters // 10) == 0:
            print(f"Iter {i:5d} | Cost: {J:.6f}")

        if np.linalg.norm(grad) < tol:
            print(f"Converged at iteration {i}")
            break

    return theta, np.array(J_history)


# === 6. Обучение модели ===
X = X_mapped.values
y = data['Passed'].values

lambda_ = 1.0
theta_reg, J_hist = gradient_descent_reg(X, y, lambda_=lambda_, alpha=0.5, num_iters=1000)

print("\nОптимальные θ (первые 5):", theta_reg[:5], "...")
print("Финальная стоимость J(θ):", J_hist[-1])

# === 7. График функции потерь ===
plt.figure(figsize=(6, 4))
plt.plot(J_hist)
plt.xlabel("Итерации")
plt.ylabel("J(θ)")
plt.title(f"Cost function with L2 regularization (λ={lambda_})")
plt.grid(True)
plt.show()


# === 8. Другие методы оптимизации (Пункт 11) ===
def scipy_optimize(X, y, lambda_, method='BFGS'):
    theta_init = np.zeros(X.shape[1])

    if method == 'Nelder-Mead':
        res = optimize.minimize(
            fun=lambda t: cost_function_reg(t, X, y, lambda_),
            x0=theta_init,
            method='Nelder-Mead',
            options={'maxiter': 20000, 'disp': True}
        )
    else:
        res = optimize.minimize(
            fun=lambda t: cost_function_reg(t, X, y, lambda_),
            x0=theta_init,
            jac=lambda t: gradient_reg(t, X, y, lambda_),
            method=method,
            options={'maxiter': 4000, 'disp': True}
        )
    return res


# === 9. Запуск BFGS ===
res_bfgs = scipy_optimize(X, y, lambda_, method='BFGS')
print("\nBFGS optimization:")
print("Success:", res_bfgs.success)
print("Final cost:", res_bfgs.fun)
theta_bfgs = res_bfgs.x

# === 10. Запуск Nelder–Mead ===
res_nm = scipy_optimize(X, y, lambda_, method='Nelder-Mead')
print("\nNelder-Mead optimization:")
print("Success:", res_nm.success)
print("Final cost:", res_nm.fun)
theta_nm = res_nm.x

# === 11. Функции предсказания вероятности и класса ===
def predict_proba_reg(theta, x1, x2, degree=6):
    """
    Возвращает вероятность прохождения контроля изделия для заданных x1 и x2.
    """
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    X_feat = map_feature(x1, x2, degree=degree).values
    probs = sigmoid(X_feat.dot(theta))
    return probs

def predict_class_reg(theta, x1, x2, degree=6, threshold=0.5):
    """
    Возвращает класс 0 или 1 в зависимости от порога threshold.
    """
    return (predict_proba_reg(theta, x1, x2, degree) >= threshold).astype(int)

# === 12. Примеры предсказаний ===
examples = np.array([
    [0.051267,0.69956],
    [0.38537,-0.56506],
    [-0.081221,1.1089]
])

for name, th in [('Gradient Descent', theta_reg),
                 ('BFGS', theta_bfgs),
                 ('Nelder-Mead', theta_nm)]:
    probs = predict_proba_reg(th, examples[:,0], examples[:,1])
    classes = predict_class_reg(th, examples[:,0], examples[:,1])
    print(f"\n{name} predictions:")
    for i, ex in enumerate(examples):
        print(f"Изделие {ex}: вероятность={probs[i]:.3f}, класс={classes[i]}")

# === 13. Построение разделяющей кривой ===
plt.figure(figsize=(8, 6))

# Сначала нарисуем исходные точки
plt.scatter(passed['Test 1'], passed['Test 2'],
            s=70, c='b', marker='+', label='Контроль пройден')
plt.scatter(failed['Test 1'], failed['Test 2'],
            s=70, c='r', marker='x', label='Контроль не пройден')

# Создаем сетку значений для x1 и x2
u = np.linspace(data['Test 1'].min()-0.1, data['Test 1'].max()+0.1, 100)
v = np.linspace(data['Test 2'].min()-0.1, data['Test 2'].max()+0.1, 100)
U, V = np.meshgrid(u, v)

# Вычисляем вероятность для каждого значения сетки
Z = predict_proba_reg(theta_bfgs, U.ravel(), V.ravel(), degree=6)
Z = Z.reshape(U.shape)

# Рисуем контур для вероятности 0.5
plt.contour(U, V, Z, levels=[0.5], colors='g', linewidths=2)
plt.xlabel('Результат теста 1')
plt.ylabel('Результат теста 2')
plt.legend()
plt.title('Разделяющая кривая (вероятность=0.5)')
plt.grid(True)
plt.show()

# Значения λ для сравнения
lambdas = [0, 1, 10, 100]

plt.figure(figsize=(16, 12))

for i, lambda_ in enumerate(lambdas, 1):
    # Обучаем модель с данным λ с помощью BFGS
    res = scipy_optimize(X, y, lambda_, method='BFGS')
    theta_opt = res.x

    # Создаем сетку для построения разделяющей кривой
    u = np.linspace(data['Test 1'].min()-0.1, data['Test 1'].max()+0.1, 100)
    v = np.linspace(data['Test 2'].min()-0.1, data['Test 2'].max()+0.1, 100)
    U, V = np.meshgrid(u, v)
    Z = predict_proba_reg(theta_opt, U.ravel(), V.ravel(), degree=6)
    Z = Z.reshape(U.shape)

    # Рисуем subplot
    plt.subplot(2, 2, i)
    plt.scatter(passed['Test 1'], passed['Test 2'],
                s=70, c='b', marker='+', label='Контроль пройден')
    plt.scatter(failed['Test 1'], failed['Test 2'],
                s=70, c='r', marker='x', label='Контроль не пройден')
    plt.contour(U, V, Z, levels=[0.5], colors='g', linewidths=2)
    plt.title(f'Разделяющая кривая (λ={lambda_})')
    plt.xlabel('Результат теста 1')
    plt.ylabel('Результат теста 2')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()