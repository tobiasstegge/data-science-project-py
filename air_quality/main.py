from read_data import read_data
from profiling import dimensionality, distribution, sparsity
from preparation import drop_columns_missing_values, drop_rows_missing_values, outlier_imputation, feature_selection

if __name__ == '__main__':
    path = './data/air_quality_tabular.csv'
    data = read_data(path)
    #dimensionality(data)
    #distribution(data)
    #sparsity(data)

    data_columns_dropped = drop_columns_missing_values(data, 0.5)
    data_rows_dropped = drop_rows_missing_values(data, 0.2)
    data_outliers_imputed = outlier_imputation(data_rows_dropped)
    data_features = feature_selection(data_outliers_imputed)
