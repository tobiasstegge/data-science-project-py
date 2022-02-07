from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame, Series
from ds_labs.ds_charts import get_variable_types, multiple_bar_chart, bar_chart
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import nan, logical_and, delete, argwhere
from matplotlib.pyplot import subplots, figure, savefig, title
from imblearn.over_sampling import SMOTE


def drop_columns_missing_values(data, threshold_factor):
    missing_values = {}
    for var in data:
        amount = data[var].isna().sum()
        if amount > 0:
            missing_values[var] = amount

    threshold = data.shape[0] * threshold_factor
    missings = [c for c in missing_values.keys() if missing_values[c] > threshold]
    print(f"Dropping columns with too many missing values on threshold {threshold_factor}", missings)
    data_dropped = data.drop(columns=missings, inplace=False)
    return data_dropped


def manual_feature_selection(data):
    columns_drop_manual = ['VEHICLE_ID', 'UNIQUE_ID', 'COLLISION_ID', 'PERSON_ID']
    print(f'Dropping columns {columns_drop_manual} in manual feature selection')
    return data.drop(columns=columns_drop_manual)


def drop_rows_missing_values(data, threshold_factor):
    print(f"Dropping rows with NaN on threshold {threshold_factor}")
    threshold = data.shape[0] * threshold_factor
    missing_values = {}
    for var in data:
        amount = data[var].isna().sum()
        if amount > 0:
            missing_values[var] = amount
    for key, value in missing_values.items():
        if value < threshold:
            print(f"Dropping missing {value} values of {key}")
            data = data[data[key].notna()]
    return data


def impute_missing_values(data):
    print("Filling missing values")
    tmp_nr, tmp_sb, tmp_bool = None, None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']

    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value=0, missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value='Unknown', missing_values=nan, copy=True)
        tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value=False, missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

    data = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
    return data


def impute_outliers(data):
    print('Imputing numerical outliers')
    mask = logical_and(data['PERSON_AGE'] > 0, data['PERSON_AGE'] < 123)
    amount = len([i for i in mask if not i])
    print(f'{amount} of PERSON_AGE values were removed')
    return data[mask]


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
    values_target, values_other = extract_target(data, target)

    train_other, test_other, train_target, test_target = train_test_split(values_other, values_target,
                                                                          train_size=train_size, stratify=values_target)

    return (concat([DataFrame(train_other, columns=data.columns.drop(target)), DataFrame(train_target, columns=[target])], axis=1),
            concat([DataFrame(test_other, columns=data.columns.drop(target)), DataFrame(test_target, columns=[target])], axis=1))


def extract_target(data, target):
    new_other = data.copy()
    return new_other.pop(target).values, new_other.values


def balancing_oversample(data, target_class, majority, minority):
    print("Balancing Target Variable (Oversampling)")
    data_majority = data[data[target_class] == majority]
    data_minority = data[data[target_class] == minority]
    data_minority_sample = DataFrame(data_minority.sample(len(data_majority), replace=True))
    return concat([data_minority_sample, data_majority], axis=0)


def balancing_undersample(data, target_class, majority, minority):
    print("Balancing Dataset (Undersampling)")
    data_majority = data[data[target_class] == majority]
    data_minority = data[data[target_class] == minority]
    data_majority_sample = DataFrame(data_majority.sample(len(data_minority), replace=True))
    return concat([data_majority_sample, data_minority], axis=0)


def balancing_smote(data, target_class):
    print("Balancing Dataset (SMOTE)")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    smote_target, smote_other = extract_target(data, target_class)
    smote_target, smote_values = smote.fit_resample(smote_other, smote_target)
    data_smote = concat([DataFrame(smote_target), DataFrame(smote_values)], axis=1)
    data_smote.columns = list(data.columns)
    return data_smote
