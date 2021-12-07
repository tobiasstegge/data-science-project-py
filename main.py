from read_data import read_data
from plot import plot
from profiling import show_dimensionality
import sys

if __name__ == '__main__':
    data = read_data(sys.argv[1])
    show_dimensionality(data)

