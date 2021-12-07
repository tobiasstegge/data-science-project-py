import pandas as pd


class Data:
    def __init__(self, file):
        self.data = pd.read_csv(file, sep=',', decimal=".", parse_dates=True)
