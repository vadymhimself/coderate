import pandas as pd
from sklearn.utils import shuffle
import os


def str_to_arr(column):
    return column.apply(lambda row: list(map(int, row[1:-1].split(', '))))

def load_data():
    data = pd.read_csv(os.path.abspath('data/bad.csv'))
    data = pd.concat([data, pd.read_csv(os.path.abspath('data/good.csv'))], ignore_index=True)
    data['encode'] = data['encode'].apply(lambda row: [int(x) for x in row.replace(r',\s$', '').split(', ')])
    # data = shuffle(data)
    # print(data.encode[0])
    return data.encode, data.Class