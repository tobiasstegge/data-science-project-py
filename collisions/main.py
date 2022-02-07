from read_data import read_data
from profiling import dimensionality, distribution, sparsity
from preparation import impute_missing_values, drop_columns_missing_values, drop_rows_missing_values, dummification, \
    impute_outliers, scaling, balancing_oversample, balancing_undersample, split_data_hold_out, extract_target, balancing_smote, \
    manual_feature_selection
from classification import naive_bayes, knn, decision_trees, random_forrest, gradient_boost, mlp
from feature_engineering import select_redundant, select_low_variance
from pattern_mining import pattern_mining, plot_top_rules, analyse_per_metric

from  mlxtend.frequent_patterns import apriori, association_rules


TARGET = 'PERSON_INJURY'
POSITIVE = 'Killed'
NEGATIVE = 'Injured'

if __name__ == '__main__':
    # Read Data
    path = './data/NYC_collisions_tabular.csv'
    data = read_data(path)

    # Profiling
    #dimensionality(data)
    #distribution(data)
    #sparsity(data)

    # Preperation
    data_dropped_manually = manual_feature_selection(data)
    data_dropped_columns = drop_columns_missing_values(data_dropped_manually, threshold_factor=0.5)
    data_dropped_rows = drop_rows_missing_values(data_dropped_columns, threshold_factor=0.1)
    data_mv_imputed = impute_missing_values(data_dropped_rows)
    data_outliers = impute_outliers(data_mv_imputed)
    data_dummified = dummification(data_outliers)
    data_scaled = scaling(data_dummified)

    # Feature Engineering
    #threshold_red = 0.98
    #data_redundant = select_redundant(data_scaled, threshold_red)

    train, test = split_data_hold_out(data_scaled, target=TARGET, positive=POSITIVE, negative=NEGATIVE, train_size=0.7)
    train_balanced = balancing_smote(train, target_class=TARGET)
    train_balanced = balancing_undersample(train, target_class=TARGET, majority=NEGATIVE, minority=POSITIVE)
    #train_sample = train_balanced.sample(int(0.1 * len(train_balanced))).copy()

    #test_target, test_other = extract_target(test, TARGET)
    #train_target, train_other = extract_target(train_sample, TARGET)

    # Classification
    #naive_bayes(train_other=train_other, test_other=test_other, train_target=train_target, test_target=test_target)
    #knn(train_other=train_other, test_other=test_other, train_target=train_target, test_target=test_target)
    #decision_trees(train_other=train_other, test_other=test_other, train_target=train_target, test_target=test_target, train=train_sample)
    #random_forrest(train_other=train_other, test_other=test_other, train_target=train_target, test_target=test_target, train=train_sample)
    #gradient_boost(train_other=train_other, test_other=test_other, train_target=train_target, test_target=test_target, train=train_sample)
    #mlp(train_other=train_other, test_other=test_other, train_target=train_target, test_target=test_target, train=train_sample)
    patterns = pattern_mining(data_scaled)
    MIN_CONF: float = 0.19
    rules = association_rules(patterns, metric='confidence', min_threshold=MIN_CONF * 5, support_only=False)
    rules.to_excel('data/müll.xlsx')

    plot_top_rules(rules, 'confidence', 'test')
    var_min_conf = [i * MIN_CONF for i in range(10, 5, -1)]
    nr_rules_cf = analyse_per_metric(rules, 'confidence', var_min_conf)
    #plot_line(var_min_conf, nr_rules_cf, title='Nr Rules x Confidence', xlabel='confidence', ylabel='Nr Rules',
    #          percentage=False)







