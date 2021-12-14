from read_data import read_data
from profiling import dimensionality, distribution, sparsity

if __name__ == '__main__':
    path = './data/air_quality_tabular.csv'
    data = read_data(path)
    #dimensionality(data)
    #distribution(data)
    sparsity(data)
