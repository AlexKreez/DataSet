import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os
import zipfile

try:
    os.system('pip install kaggle')
    os.system('kaggle datasets download -d fivethirtyeight/fivethirtyeight-comic-characters-dataset')
except:
    print('')

with zipfile.ZipFile('fivethirtyeight-comic-characters-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()

data = pd.read_csv('dc-wikia-data.csv')

print("Размер данных:", data.shape)

print("Типы колонок:\n", data.dtypes)

missing_values = data.isnull().sum()
print("Пропущенные значения:\n", missing_values[missing_values > 0])

num_cols = []
for col in data.columns:
    missings_count = data[data[col].isnull()].shape[0]
    dt = str(data[col].dtype)
    if missings_count > 0 and (dt == 'float64' or dt == 'int64'):
        num_cols.append(col)
        print(f'Колонка {col}. Тип данных {dt}. Количество пропущенных значений {missings_count}')
        imp_num = SimpleImputer(strategy='mean')
        data[num_cols] = imp_num.fit_transform(data[num_cols])

obj_cols = []
for col in data.columns:
    missings_count = data[data[col].isnull()].shape[0]
    dt = str(data[col].dtype)
    if col == 'GSM':
        data['GSM'] = data['GSM'].fillna('Norm')
    if missings_count > 0 and dt == 'object':
        obj_cols.append(col)
        print(f'Колонка {col}. Тип данных {dt}. Количество пропущенных значений {missings_count} .')
        imp_obj = SimpleImputer(strategy='most_frequent')
        data[obj_cols] = imp_obj.fit_transform(data[obj_cols])

print("Пропущенные значения после обработки:\n", data.isnull().sum().sum())

data.to_csv('dc-wikia-data-processed.csv', index=False)
print("Обработанные данные сохранены в 'dc-wikia-data-processed.csv'")
