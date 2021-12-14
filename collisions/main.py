from read_data import read_data
from profiling import dimensionality, distribution

if __name__ == '__main__':
    path = './data/NYC_collisions_tabular.csv'
    data = read_data(path)
    dimensionality(data)
    distribution(data)
