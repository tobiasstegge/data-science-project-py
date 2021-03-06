import pandas as pd


def read_data(file):
    print("Reading Data")
    data = pd.read_csv(file, index_col='DATETIME', sep=',', decimal='.', parse_dates={
        'DATETIME': ['CRASH_DATE', 'CRASH_TIME']})

    print("Transforming Data Types")
    # transform object into category type
    cat_vars = data.select_dtypes(include='object')
    data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

    return data


class Data:
    def __init__(self, file):
        pass
