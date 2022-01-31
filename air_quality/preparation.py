from seaborn import heatmap
from matplotlib.pyplot import figure, title, savefig
from ds_labs.ds_charts import get_variable_types, multiple_bar_chart, bar_chart
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame, concat
from matplotlib.pyplot import subplots
from numpy import delete, argwhere


def drop_columns_missing_values(data, threshold_factor):
    print(f"Dropping columns which have more than {threshold_factor * 100}% of missing values")
    missing_values = {}
    for var in data:
        amount = data[var].isna().sum()
        if amount > 0:
            missing_values[var] = amount

    threshold = data.shape[0] * (1 - threshold_factor)
    missings = [c for c in missing_values.keys() if missing_values[c] > threshold]
    data_dropped = data.drop(columns=missings, inplace=False)
    print('Dropped variables: ', missings)
    return data_dropped


def drop_rows_missing_values(data, threshold_factor):
    print(f"Dropping all rows with missing values")
    data_dropped = data.dropna()
    print(f"Removed {(1 - len(data_dropped) / len(data)) * 100}% of rows")
    return data_dropped


def outlier_imputation(data):
    print('Imputing numerical outliers')
    mask = (data['CO_Mean'] < 10) & (data['CO_Max'] < 20) & \
           (data['CO_Std'] < 10) & (data['NO2_Max'] < 250) & (data['NO2_Std'] < 80) & (data['O3_Max'] < 400) & \
           (data['O3_Std'] < 125) & (data['PM2.5_Mean'] < 1000) & (data['PM2.5_Max'] < 3000) & \
           (data['PM2.5_Std'] < 750) & (data['PM2.5_Mean'] < 6000) & (data['PM10_Max'] < 6000) & \
           (data['PM10_Std'] < 2000) & (data['SO2_Max'] < 500) & (data['SO2_Std'] < 150)
    data_outliers_removed = data[mask]
    print(f'Classified and removed {((len(data) - len(data_outliers_removed)) / len(data)) * 100}% or'
          f' {len(data) - len(data_outliers_removed)} of outliers')
    return data_outliers_removed


def feature_selection(data, threshold):
    # manual dropping of columns
    columns_to_drop = ['FID', 'City_EN', 'Prov_EN', 'GbProv', 'GbCity', 'Field_1']
    print(f'Dropping columns {columns_to_drop} for feature selection')
    data_manual = data.drop(columns=columns_to_drop)

    # TODO

    # dropping redundant columns
    correlation_mtx = abs(data_manual.corr())
    vars_2_drop = {}
    for column in correlation_mtx.columns:
        columns_corr = correlation_mtx[column].loc[correlation_mtx[column] >= threshold]
        if column not in vars_2_drop and (len(columns_corr) > 1):
            vars_2_drop[column] = columns_corr.index.values

    selected_2_drop = []
    print(vars_2_drop.keys())
    for key in vars_2_drop.keys():
        if key not in selected_2_drop:
            for var in vars_2_drop[key]:
                if var != key and var not in selected_2_drop:
                    selected_2_drop.append(var)
    print('Variables selected to drop', selected_2_drop)

    # print new correlation matrix
    correlation_mtx.drop(labels=vars_2_drop, axis=1, inplace=True)
    correlation_mtx.drop(labels=vars_2_drop, axis=0, inplace=True)
    figure(figsize=[10, 10])
    heatmap(correlation_mtx, xticklabels=correlation_mtx.columns, yticklabels=correlation_mtx.columns, cmap='Blues', annot=True)
    title('Filtered Correlation Analysis')
    savefig(f'images/filtered_heatmaps_feature_selection_{threshold}.png')

    # checking low variance
    threshold = 0.2
    lst_variables = []
    lst_variances = []

    variable_types = get_variable_types(data_manual)
    numeric_vars = variable_types['Numeric']
    data_manual_num = data_manual[numeric_vars]
    for el in data_manual_num.columns:
        value = data[el].var()
        if value >= threshold:
            lst_variables.append(el)
            lst_variances.append(value)
    print(len(lst_variables), lst_variables)
    figure(figsize=[10, 4])
    bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance')
    savefig('images/filtered_variance_analysis.png')

    return data_manual.drop(columns=selected_2_drop)


def scaling(data, plot_figure=True):
    print("Scaling numeric variables")
    variable_types = get_variable_types(data)

    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']
    data_numeric = data[numeric_vars]
    data_symbolic = data[symbolic_vars]
    data_bool = data[boolean_vars]

    standard_scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data_numeric)
    tmp_zscore = DataFrame(standard_scaler.transform(data_numeric), index=data.index, columns=numeric_vars)
    norm_data_zscore = concat([tmp_zscore, data_symbolic, data_bool], axis=1)

    min_max_scaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data_numeric)
    tmp_min_max = DataFrame(min_max_scaler.transform(data_numeric), index=data.index, columns=numeric_vars)

    if plot_figure:
        fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
        axs[0, 0].set_title('Original data')
        data_numeric.boxplot(ax=axs[0, 0])
        axs[0, 1].set_title('Z-score normalization')
        tmp_zscore.boxplot(ax=axs[0, 1])
        axs[0, 2].set_title('MinMax normalization')
        tmp_min_max.boxplot(ax=axs[0, 2])

        savefig('images/scaling.png')

    return norm_data_zscore


def split_data(data, target, positive, negative, train_size):
    print(f'Splitting for train & test data using hold-out for target {target}')
    values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

    values_target = data.pop(target).values
    values_other = data.values

    train_other, test_other, train_target, test_target = train_test_split(values_other, values_target,
                                                                          train_size=train_size, stratify=values_target)

    values['Train'] = [len(delete(train_target, argwhere(train_target == negative))),
                       len(delete(train_target, argwhere(train_target == positive)))]
    values['Test'] = [len(delete(test_target, argwhere(test_target == negative))),
                      len(delete(test_target, argwhere(test_target == positive)))]

    figure(figsize=(12, 4))
    multiple_bar_chart([positive, negative], values, title='Data distribution per class variable')
    savefig('images/split_train_test.png')

    train = concat([DataFrame(train_other, columns=data.columns), DataFrame(train_target, columns=[target])], axis=1)
    test = concat([DataFrame(test_other, columns=data.columns), DataFrame(test_target, columns=[target])], axis=1)
    return train, test


def balancing_undersample(data, target, positive, negative):
    print("Balancing Target Variable (Undersampling)")
    data_majority = data[data[target] == positive]
    data_minority = data[data[target] == negative]
    data_majority_sample = DataFrame(data_majority.sample(len(data_minority), replace=True))
    return concat([data_majority_sample, data_minority], axis=0)
