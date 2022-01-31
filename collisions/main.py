from read_data import read_data
from profiling import dimensionality, distribution, sparsity
from preparation import impute_missing_values, drop_columns_missing_values, drop_rows_missing_values, dummification, \
    impute_outliers, scaling, balancing_oversample, balancing_undersample, split_data, extract_target, balancing_smote, \
    feature_selection
from classification import naive_bayes

TARGET = 'PERSON_INJURY'
POSITIVE = 'Killed'
NEGATIVE = 'Injured'

if __name__ == '__main__':
    # Read Data
    path = './data/NYC_collisions_tabular.csv'
    data = read_data(path)

    # Profiling
    # dimensionality(data)
    # distribution(data)
    # sparsity(data)

    # Preperation
    data_dropped_columns = drop_columns_missing_values(data, threshold_factor=0.9)
    data_dropped_rows = drop_rows_missing_values(data_dropped_columns, threshold_factor=0.1)
    data_mv_impted = impute_missing_values(data_dropped_rows)
    data_outliers = impute_outliers(data_dropped_rows)
    data_feature_selection = feature_selection(data_outliers)
    data_dummified = dummification(data_feature_selection)
    data_scaled = scaling(data_dummified)
    train, test = split_data(data_scaled, target=TARGET, positive=POSITIVE, negative=NEGATIVE, train_size=0.7)

    balanced_data = {}

    balanced_data['Oversample'] = balancing_oversample(train, target_class=TARGET, majority=NEGATIVE, minority=POSITIVE)
    balanced_data['Undersample'] = balancing_undersample(train, target_class=TARGET, majority=NEGATIVE, minority=POSITIVE)

    # Classification
    test_target, test_other = extract_target(test, TARGET)
    for key, data_train in balanced_data.items():
        train_target, train_other = extract_target(data_train, TARGET)
        naive_bayes(train_other=train_other, test_other=test_other, train_target=train_target, test_target=test_target,
                    balancing_type=key)
