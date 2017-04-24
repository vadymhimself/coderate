import pandas as pd
from sklearn.utils import shuffle
import os


def str_to_arr(column):
    return column.apply(lambda row: list(map(int, row[1:-1].split(', '))))

def load_data():
    data = pd.read_csv(os.path.abspath('data/final_13.csv'))
    data['encode'] = data['encode'].apply(lambda row: [int(c) for c in row[1:-1].split(', ')])
    # print(data.encode)
    return data.encode, data.Class