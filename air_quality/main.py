from read_data import read_data
from pandas import read_csv
from profiling import dimensionality, distribution, sparsity
from preparation import drop_columns_missing_values, drop_rows_missing_values, outlier_imputation, feature_selection, \
    scaling, split_data, balancing_undersample
from classification import naive_bayes, knn, decision_trees
from evaluation import roc_chart

TARGET = 'ALARM'
POSITIVE = 'Safe'
NEGATIVE = 'Danger'

if __name__ == '__main__':
    path = './data/air_quality_tabular.csv'
    data = read_data(path)
    # dimensionality(data)
    # distribution(data)
    # sparsity(data)

    data_columns_dropped = drop_columns_missing_values(data, 0.5)
    data_rows_dropped = drop_rows_missing_values(data_columns_dropped, 0.2)
    data_outliers_imputed = outlier_imputation(data_rows_dropped)
    data_feature_selection = feature_selection(data_outliers_imputed, threshold=0.9)
    data_scaled = scaling(data_feature_selection)

    train, test = split_data(data_scaled, target=TARGET, positive=POSITIVE, negative=NEGATIVE, train_size=0.7)
    train_balanced = balancing_undersample(train, target=TARGET, positive=POSITIVE, negative=NEGATIVE)

    train_sample = train_balanced.sample(int(0.2 * len(train_balanced)))

    train_target = train_sample.pop(TARGET).values
    train_other = train_sample.values
    test_target = test.pop(TARGET).values
    test_other = test.values

    #nb_model = naive_bayes(train_other=train_other, train_target=train_target, test_target=test_target,
    #                       test_other=test_other)

    knn = knn(train_other=train_other, train_target=train_target, test_target=test_target, test_other=test_other)
    #tree = decision_trees(train_other=train_other, train_target=train_target, test_target=test_target,
    #                      test_other=test_other)

    # roc(model=nb_model, test_other=test_other, test_target=test_target, target=TARGET)
