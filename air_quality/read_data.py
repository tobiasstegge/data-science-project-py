import pandas as pd


def read_data(file):
    data = pd.read_csv(file, index_col='date', sep=',', decimal='.')

    # transform object into category type
    cat_vars = data.select_dtypes(include='object')
    data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

    return data


class Data:
    def __init__(self, file):
        pass
