import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import re
from nltk.stem import PorterStemmer

# === 1. Загрузка данных ex5data1.mat ===
data = loadmat('ex5data1.mat')
X = data['X']
y = data['y'].ravel()

# === 2. Визуализация данных ===
def plot_data(X, y):
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='b', marker='+', label='y=1')
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='o', label='y=0')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()

plt.figure(figsize=(7,5))
plot_data(X, y)
plt.title('Линейно разделимые данные (ex5data1.mat)')
plt.show()

# === 3–4. Обучение SVM с линейным ядром и построение разделяющих прямых ===
def train_svm_and_get_line(X, y, C_value):
    """Обучает SVM с линейным ядром и возвращает параметры разделяющей прямой."""
    model = svm.SVC(C=C_value, kernel='linear')
    model.fit(X, y)
    w = model.coef_[0]
    b = model.intercept_[0]
    return model, w, b

# Обучаем для C=1 и C=100
model_C1, w1, b1 = train_svm_and_get_line(X, y, 1)
model_C100, w100, b100 = train_svm_and_get_line(X, y, 100)

# Строим общий график
plt.figure(figsize=(8,6))
plot_data(X, y)

x_plot = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_plot_C1 = -(w1[0]/w1[1])*x_plot - b1/w1[1]
y_plot_C100 = -(w100[0]/w100[1])*x_plot - b100/w100[1]

plt.plot(x_plot, y_plot_C1, 'g--', label='Разделяющая прямая (C=1)')
plt.plot(x_plot, y_plot_C100, 'k-', label='Разделяющая прямая (C=100)')
plt.title('Сравнение разделяющих прямых при разных C')
plt.legend()
plt.show()

# === 5. Реализация функции вычисления Гауссового ядра ===
def gaussian_kernel(x1, x2, sigma):
    """Вычисляет значение Гауссового (RBF) ядра между двумя векторами x1 и x2."""
    diff = x1 - x2
    return np.exp(-(diff @ diff) / (2 * (sigma ** 2)))

# Пример проверки функции:
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)
print(f"Пример вычисления Гауссового ядра: {sim:.4f}")

print("""
Пояснение:
- При C=1 SVM допускает некоторые ошибки классификации → граница мягче, обобщающая.
- При C=100 ошибки штрафуются сильнее → граница становится жёсткой и ближе к данным (риск переобучения).
""")

# === 6. Загрузка данных ex5data2.mat ===
data2 = loadmat('ex5data2.mat')
X2 = data2['X']
y2 = data2['y'].ravel()

plt.figure(figsize=(7,5))
plot_data(X2, y2)
plt.title('Нелинейно разделимые данные (ex5data2.mat)')
plt.show()

# === 7. Обработка данных с помощью функции Гауссового ядра ===
sigma = 0.1

print("\nПримеры вычисления Гауссового ядра между первыми точками:")
for i in range(3):
    sim = gaussian_kernel(X2[0], X2[i+1], sigma)
    print(f"Сходство между точкой 0 и {i+1}: {sim:.4f}")

print("\nЧем дальше точки друг от друга, тем меньше значение ядра (меньше сходство).")

# === 8. Обучение SVM с Гауссовым (RBF) ядром ===
C_value = 1
gamma = 1 / (2 * sigma**2)  # параметр γ = 1/(2σ²)

model_rbf = svm.SVC(C=C_value, kernel='rbf', gamma=gamma)
model_rbf.fit(X2, y2)

# === 9. Визуализация разделяющей границы ===
def plot_decision_boundary(model, X, y):
    plot_data(X, y)

    # создаём сетку точек для предсказаний
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Рисуем границу (уровень 0.5 между классами)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.5)
    plt.title(f'SVM с Гауссовым ядром (C={C_value}, σ={sigma})')
    plt.show()

plot_decision_boundary(model_rbf, X2, y2)

# === 10. Загрузка данных ex5data3.mat ===
data3 = loadmat('ex5data3.mat')
X3 = data3['X']
y3 = data3['y'].ravel()
Xval = data3['Xval']
yval = data3['yval'].ravel()

# Визуализация исходных данных
def plot_data(X, y):
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='b', marker='+', label='y=1')
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='o', label='y=0')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()

plt.figure(figsize=(7,5))
plot_data(X3, y3)
plt.title('Обучающие данные (ex5data3.mat)')
plt.show()

# === 11. Подбор параметров C и σ на валидационной выборке ===
def dataset3_params(X, y, Xval, yval):
    """Подбирает лучшие C и σ, минимизирующие ошибку на валидационной выборке."""
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    best_C = 0
    best_sigma = 0
    best_error = float('inf')

    for C in C_values:
        for sigma in sigma_values:
            gamma = 1 / (2 * sigma**2)
            model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            model.fit(X, y)

            preds = model.predict(Xval)
            error = np.mean(preds != yval)

            if error < best_error:
                best_error = error
                best_C = C
                best_sigma = sigma

    return best_C, best_sigma, best_error

best_C, best_sigma, best_error = dataset3_params(X3, y3, Xval, yval)
print(f"\nОптимальные параметры:\nC = {best_C}, σ = {best_sigma}, ошибка = {best_error:.4f}")

# === 12. Обучение SVM с найденными параметрами и визуализация ===
def plot_decision_boundary(model, X, y, title):
    plot_data(X, y)
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.5)
    plt.title(title)
    plt.show()

# Обучаем SVM с найденными параметрами
gamma = 1 / (2 * best_sigma**2)
best_model = svm.SVC(C=best_C, kernel='rbf', gamma=gamma)
best_model.fit(X3, y3)

plot_decision_boundary(best_model, X3, y3,
                       f"SVM с оптимальными параметрами (C={best_C}, σ={best_sigma})")

