from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from ds_labs.ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from numpy import nan, logical_and
from matplotlib.pyplot import subplots, savefig

def drop_columns_missing_values(data, threshold_factor):
    print(f"Dropping columns with too many missing values on threshold {threshold_factor}")
    missing_values = {}
    for var in data:
        amount = data[var].isna().sum()
        if amount > 0:
            missing_values[var] = amount

    threshold = data.shape[0] * threshold_factor
    missings = [c for c in missing_values.keys() if missing_values[c] > threshold]
    data_dropped = data.drop(columns=missings, inplace=False)
    print('Dropped variables: ', missings)
    return data_dropped


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
    data.to_csv(f'data/drop_rows_mv.csv', index=False)
    return data


def impute_missing_values(data):
    print("Filling missing values")
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']

    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value=0, missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=nan, copy=True)
        tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value=False, missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

    data = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
    data.to_csv('data/mv_imputation.csv', index=False)
    return data


def impute_outliers(data):
    print('Imputing numerical outliers')
    mask = logical_and(data['PERSON_AGE'] > 0, data['PERSON_AGE'] < 123)
    return data[mask]


def dummification(data):
    print('Dummify symbolic variables')
    symbolic_vars = get_variable_types(data)['Symbolic']
    symbolic_vars.remove('PERSON_ID')
    other_vars = [c for c in data.columns if c not in symbolic_vars]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    dataset_symbolic = data[symbolic_vars]
    encoder.fit(dataset_symbolic)
    new_vars = encoder.get_feature_names(symbolic_vars)
    transformed_dataset = encoder.transform(dataset_symbolic)
    dummy = DataFrame(transformed_dataset, columns=new_vars, index=dataset_symbolic.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)
    final_data = concat([data[other_vars], dummy], axis=1)
    final_data.to_csv(f'data/dummified.csv', index=False)
    print(f"Column size after Dummification: {final_data.columns.size}")
    return final_data


def scaling(data):
    print("Scaling numeric variables")
    variable_types = get_variable_types(data)
    numeric_vars = ['PERSON_AGE']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']
    data_numeric = data[numeric_vars]
    data_symbolic = data[symbolic_vars]
    data_bool = data[boolean_vars]

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data_numeric)
    tmp_standard = DataFrame(transf.transform(data_numeric), index=data.index, columns=numeric_vars)
    norm_data_zscore = concat([tmp_standard, data_symbolic, data_bool], axis=1)
    norm_data_zscore.to_csv(f'data/scaled_zscore.csv', index=False)
    return norm_data_zscore


def balancing(data):
    print("Balancing Target Variable (Oversampling)")
    data_majority = data[data["PERSON_INJURY"] == 'Injured']
    data_minority = data[data["PERSON_INJURY"] == 'Killed']
    data_minority_sample = DataFrame(data_minority.sample(len(data_majority), replace=True))
    data_oversampling = concat([data_minority_sample, data_majority], axis=0)
    data_oversampling.to_csv(f'data/oversampling.csv', index=False)
    return data_oversampling
