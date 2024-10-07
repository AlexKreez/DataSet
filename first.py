import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv('train.csv')

# Преобразование дат
data['Order Date'] = pd.to_datetime(data["Order Date"], format="%d/%m/%Y")
data['Ship Date'] = pd.to_datetime(data['Ship Date'], format="%d/%m/%Y")

# Проверка пропущенных значений
missing_values = data.isnull().sum()
print("Пропущенные значения до очистки:\n", missing_values)
# Очистка данных
data = data.dropna(subset=['Postal Code'])

# Описательная статистика
description = data.describe()


# Распределение данных
plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], bins=50, kde=True)
plt.title('Распределение продаж')
plt.xlabel('Продажи')
plt.ylabel('Частота')
plt.show()


# Проверка выбросов
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Sales'])
plt.title('Выбросы в продажах')
plt.xlabel('Продажи')
plt.show()

# Визуализация зависимостей
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='Order Date', y='Sales', hue='Category')
plt.title('Зависимость продаж от даты заказа')
plt.xlabel('Дата заказа')
plt.ylabel('Продажи')
plt.legend(title='Категория')
plt.show()


data['Month'] = data['Order Date'].dt.month
data['Year'] = data['Order Date'].dt.year
monthly_sales = data.groupby(['Year', 'Month']).size().reset_index(name='Sales Count')

plt.figure(figsize=(12, 8))
sns.lineplot(data=monthly_sales, x='Month', y='Sales Count', hue='Year', marker='o')

plt.title('Количество ежемесячных продаж за несколько лет')
plt.xlabel('Месяц')
plt.ylabel('Количкство продаж')
plt.legend(title='год')
plt.show()