from read_data import read_data
from profiling import dimensionality, distribution, sparsity
from preparation import impute_missing_values, drop_columns_missing_values, drop_rows_missing_values, dummification, \
    impute_outliers, scaling, balancing_oversample, balancing_undersample
from classification import split_data
from evaluation import plot_roc_chart, create_confusion_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # Read Data
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
    data_balanced_oversample = balancing_oversample(data_scaled)
    data_balanced_undersample = balancing_undersample(data_scaled)

    # classification
    train_other, test_other, train_target, test_target = split_data(data_balanced_undersample, target='PERSON_INJURY',
                                                                    positive='Injured', negative='Killed')

    model_nb = GaussianNB().fit(train_other, train_target)
    model_nb_cat = BernoulliNB().fit(train_other, train_target)

    model_reg = LogisticRegression(max_iter=1000)
    model_reg.fit(train_other, train_target)
    # create_confusion_matrix(model, test_target, test_other)
    plot_roc_chart({'Logistic_Regression': model_reg, 'NB': model_nb, 'Cat_NB:': model_nb_cat}, test_other=test_other, test_target=test_target,
                   target='PERSON_INJURY',
                   ax=None)
