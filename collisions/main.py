from read_data import read_data
from profiling import dimensionality, distribution, sparsity
from preparation import impute_missing_values, drop_columns_missing_values, drop_rows_missing_values, dummification, \
    impute_outliers, scaling, balancing

if __name__ == '__main__':
    path = './data/NYC_collisions_tabular.csv'
    data = read_data(path)
    # Profiling
    #dimensionality(data)
    #distribution(data)
    #sparsity(data)

    # Preperation
    data_dropped_columns = drop_columns_missing_values(data, threshold_factor=0.9)
    data_dropped_rows = drop_rows_missing_values(data_dropped_columns, threshold_factor=0.1)
    data_mv = impute_missing_values(data_dropped_rows)
    data_outliers = impute_outliers(data_mv)
    data_dummified = dummification(data_outliers)
    data_scaled = scaling(data_dummified)
    balancing(data_scaled)


