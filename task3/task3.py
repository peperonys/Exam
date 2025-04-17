import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Генерация данных
np.random.seed(0)
n_samples = 100
price = np.random.uniform(10, 100, n_samples)  # Цена товара
advertising = np.random.uniform(1000, 5000, n_samples)  # Расходы на рекламу
seasonality = np.random.choice([0, 1], n_samples)  # Сезонный фактор (0 - вне сезона, 1 - в сезоне)
demand = 50 + 2 * price + 0.5 * advertising + 10 * seasonality + np.random.normal(0, 10, n_samples)  # Спрос

# Создание DataFrame
data = pd.DataFrame({
    'Price': price,
    'Advertising': advertising,
    'Seasonality': seasonality,
    'Demand': demand
})

# Разделение данных на обучающую и тестовую выборки
X = data[['Price', 'Advertising', 'Seasonality']]
y = data['Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse:.2f}")

# Визуализация результатов
plt.scatter(y_test, y_pred)
plt.xlabel('Фактический спрос')
plt.ylabel('Прогнозируемый спрос')
plt.title('Фактический vs Прогнозируемый спрос')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')  # Линия y=x
plt.show()

# Вывод коэффициентов
print("Коэффициенты модели:")
print(f"Свободный член (β0): {model.intercept_:.2f}")
print(f"Коэффициент при цене (β1): {model.coef_[0]:.2f}")
print(f"Коэффициент при рекламе (β2): {model.coef_[1]:.2f}")
print(f"Коэффициент при сезонности (β3): {model.coef_[2]:.2f}")