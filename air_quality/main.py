from read_data import read_data
from profiling import dimensionality, distribution, sparsity
from preparation import drop_columns_missing_values, drop_rows_missing_values, outlier_imputation, scaling, \
    split_data_hold_out, balancing_undersample, manual_feature_selection, extract_target, dummification
from classification import naive_bayes, knn, decision_trees, random_forrest, gradient_boost, mlp
from feature_engineering import select_redundant, select_low_variance
from ds_labs.ds_charts import get_variable_types
from sklearn.preprocessing import KBinsDiscretizer
from pandas import DataFrame
from pattern_mining import pattern_mining, plot_line
from  mlxtend.frequent_patterns import association_rules

TARGET = 'ALARM'
POSITIVE = 'Danger'
NEGATIVE = 'Safe'

if __name__ == '__main__':
    path = './data/air_quality_tabular.csv'
    data = read_data(path)
    # dimensionality(data)
    # distribution(data)
    # sparsity(data)

    # Preparation
    data_manual_feature_selection = manual_feature_selection(data)
    data_columns_dropped = drop_columns_missing_values(data_manual_feature_selection, 0.5)
    data_rows_dropped = drop_rows_missing_values(data_columns_dropped)
    data_outliers_imputed = outlier_imputation(data_rows_dropped)

    data_scaled = scaling(data_outliers_imputed)

    #data_outliers_imputed.to_csv(f'data/after_preparation_air.csv', index=False)

    len_before = len(data_outliers_imputed.columns)
    # feature engineering
    threshold_correlation = 0.8
    threshold_variance = 0
    #data_correlations = select_redundant(data_outliers_imputed, threshold_correlation=threshold_correlation)
    #data_variance = select_low_variance(data_correlations, threshold_variance=threshold_variance)

    # Classification
    #train, test = split_data_hold_out(data_outliers_imputed, target=TARGET, positive=POSITIVE, negative=NEGATIVE, train_size=0.7)
    #train_balanced = balancing_undersample(train, target_class=TARGET, majority=NEGATIVE, minority=POSITIVE)
    #train_sample = train_balanced.sample(int(0.2 * len(train_balanced)))

    #train_target, train_other = extract_target(train_sample, target=TARGET)
    #test_target, test_other = extract_target(test, target=TARGET)

    #nb = naive_bayes(train_other=train_other, train_target=train_target, test_target=test_target,
    #                 test_other=test_other)

    #print(f'{threshold_variance} {abs(len(data_correlations.columns) - len_before)}')
    #knn = knn(train_other=train_other, train_target=train_target, test_target=test_target, test_other=test_other, t=threshold_variance)

    #tree = decision_trees(train_other=train_other, train_target=train_target, test_target=test_target,
    #                      test_other=test_other, train=train_sample)

    #forrest = random_forrest(train_other=train_other, train_target=train_target, test_target=test_target, test_other=test_other, train=train_sample)

    #gradient_boost(train_other=train_other, train_target=train_target, test_target=test_target, test_other=test_other, train=train_sample)

    #mlp = mlp(train_other=train_other, train_target=train_target, test_target=test_target, test_other=test_other, train=train_sample)

    # roc(model=nb_model, test_other=test_other, test_target=test_target, target=TARGET)


    numeric_vars = get_variable_types(data_scaled)['Numeric']
    numeric_vars_means = [var for var in numeric_vars if var[-4:] == 'Mean']

    data_means = data[numeric_vars_means]
    data_means.dropna(inplace=True)
    kbins = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='uniform')
    data_trans = kbins.fit_transform(data_means)

    frame = DataFrame(data_trans)
    frame.columns = kbins.get_feature_names_out()

    patterns = pattern_mining(frame)
    MIN_CONF: float = 0.19
    rules = association_rules(patterns, metric='confidence', min_threshold=MIN_CONF * 5, support_only=False)
    rules.to_excel('data/müll2.xlsx')




