from read_data import read_data
from plot import plot, stem
from profiling import dimensionality, distribution
import sys

if __name__ == '__main__':
    data = read_data(sys.argv[1])
    dimensionality(data)
    distribution(data)
