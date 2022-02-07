from seaborn import heatmap
from matplotlib.pyplot import figure, title, savefig
from ds_labs.ds_charts import get_variable_types, multiple_bar_chart, bar_chart
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from pandas import DataFrame, concat
from matplotlib.pyplot import subplots
from numpy import delete, argwhere
from imblearn.over_sampling import SMOTE


def manual_feature_selection(data):
    columns_to_drop = ['FID', 'City_EN', 'Prov_EN', 'GbProv', 'GbCity', 'Field_1']
    print(f'Dropping columns {columns_to_drop} for feature selection')
    return data.drop(columns=columns_to_drop)


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


def drop_rows_missing_values(data):
    data_dropped = data.dropna()
    print(f"Removed {(1 - len(data_dropped) / len(data)) * 100}% of rows with missing values")
    return data_dropped


def outlier_imputation(data):
    print('Imputing numerical outliers')
    mask = (data['CO_Mean'] < 10) & (data['CO_Max'] < 20) & \
           (data['CO_Std'] < 10) & (data['NO2_Max'] < 250) & (data['NO2_Std'] < 80) & (data['O3_Max'] < 400) & \
           (data['O3_Std'] < 125) & (data['PM2.5_Mean'] < 1000) & (data['PM2.5_Max'] < 3000) & \
           (data['PM2.5_Std'] < 750) & (data['PM2.5_Mean'] < 6000) & (data['PM10_Max'] < 6000) & \
           (data['PM10_Std'] < 2000) & (data['SO2_Max'] < 500) & (data['SO2_Std'] < 150)
    data_outliers_removed = data[mask]
    print(f'Manually selected and removed {((len(data) - len(data_outliers_removed)) / len(data)) * 100}% or'
          f' {len(data) - len(data_outliers_removed)} of outliers')
    return data_outliers_removed


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
    norm_data_min_max = concat([tmp_min_max, data_symbolic, data_bool], axis=1)

    if plot_figure:
        fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
        axs[0, 0].set_title('Original data')
        data_numeric.boxplot(ax=axs[0, 0])
        axs[0, 1].set_title('Z-score normalization')
        tmp_zscore.boxplot(ax=axs[0, 1])
        axs[0, 2].set_title('MinMax normalization')
        tmp_min_max.boxplot(ax=axs[0, 2])

        savefig('images/scaling.png')

    return norm_data_min_max


def split_data_hold_out(data, target, positive, negative, train_size):
    print(f'Splitting for train & test data using hold-out for target {target}')
    all_data = data
    values = {'Original': [len(all_data[all_data[target] == positive]), len(all_data[all_data[target] == negative])]}
    values_target = all_data.pop(target).values
    values_other = all_data.values

    train_other, test_other, train_target, test_target = train_test_split(values_other, values_target,
                                                                          train_size=train_size, stratify=values_target)

    train = concat([DataFrame(train_other, columns=all_data.columns), DataFrame(train_target, columns=[target])], axis=1)
    test = concat([DataFrame(test_other, columns=all_data.columns), DataFrame(test_target, columns=[target])], axis=1)

    # graphing split
    values['Train'] = [len(delete(train_target, argwhere(train_target == negative))),
                       len(delete(train_target, argwhere(train_target == positive)))]
    values['Test'] = [len(delete(test_target, argwhere(test_target == negative))),
                      len(delete(test_target, argwhere(test_target == positive)))]

    figure(figsize=(12, 4))
    multiple_bar_chart([positive, negative], values, title='Data distribution per class variable')
    savefig('images/split_train_test.png')

    return train, test


def split_data_kfold_cross(data, target, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)
    target, others = extract_target(data=data, target=target)
    for train_idx, test_idx in skf.split(others, target):
        train_other, test_other = others[train_idx], others[test_idx]
        train_target, test_target = target[train_idx], target[test_idx]
        # TODO


def extract_target(data, target):
    all_data = data
    target_data = all_data.pop(target).values
    other_data = all_data.values
    return target_data, other_data


def balancing_oversample(data, target_class, majority, minority):
    print("Balancing Target Variable (Oversampling)")
    data_majority = data[data[target_class] == majority]
    data_minority = data[data[target_class] == minority]
    data_minority_sample = DataFrame(data_minority.sample(len(data_majority), replace=True))
    return concat([data_minority_sample, data_majority], axis=0)


def balancing_undersample(data, target_class, majority, minority):
    print("Balancing Dataset (Undersampling)")
    all_data = data
    data_majority = all_data[all_data[target_class] == majority]
    data_minority = all_data[all_data[target_class] == minority]
    data_majority_sample = DataFrame(data_majority.sample(len(data_minority), replace=True))
    return concat([data_majority_sample, data_minority], axis=0)


def balancing_smote(data, target_class):
    print("Balancing Dataset (SMOTE)")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    target, other = extract_target(data, target_class)
    other = data.values
    smote_target, smote_values = smote.fit_resample(other, target)
    data_smote = concat([DataFrame(smote_target), DataFrame(smote_values)], axis=1)
    data_smote.columns = list(data.columns) + [target_class]
    return data_smote


def dummification(data):
    print('Dummify symbolic variables')
    len_before = data.columns.size
    symbolic_vars = get_variable_types(data)['Symbolic']
    other_vars = [c for c in data.columns if c not in symbolic_vars]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    dataset_symbolic = data[symbolic_vars]
    encoder.fit(dataset_symbolic)
    new_vars = encoder.get_feature_names(symbolic_vars)
    transformed_dataset = encoder.transform(dataset_symbolic)
    dummy = DataFrame(transformed_dataset, columns=new_vars, index=dataset_symbolic.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)
    final_data = concat([data[other_vars], dummy], axis=1)
    print(f"Column size after Dummification from {len_before} to {final_data.columns.size}")
    return final_data

