from seaborn import heatmap
from matplotlib.pyplot import figure, title, savefig

THRESHOLD = 0.9

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
    print(f"Dropping rows which have less than {threshold_factor * 100}% of missing values")
    data_dropped = data.dropna()
    data_dropped.to_csv(f'data/drop_rows_mv.csv', index=False)
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


def feature_selection(data):
    # manual dropping of columns
    data_manual = data.drop(columns=['FID', 'City_EN', 'Prov_EN', 'GbProv', 'GbCity', 'Field_1'])
    # dropping redundant columns
    correlation_mtx = abs(data_manual.corr())
    vars_2_drop = []
    for column in correlation_mtx.columns:
        columns_corr = correlation_mtx[column].loc[correlation_mtx[column] >= THRESHOLD]
        columns_corr = columns_corr.drop(column)
        if len(columns_corr) == 0:
            correlation_mtx.drop(labels=column, axis=1, inplace=True)
            correlation_mtx.drop(labels=column, axis=0, inplace=True)
        elif column not in vars_2_drop:
            vars_2_drop.append(columns_corr.index.all())

    print(f'Dropping variables {vars_2_drop} on threshold {THRESHOLD}')
    figure(figsize=[10, 10])
    heatmap(correlation_mtx, xticklabels=correlation_mtx.columns, yticklabels=correlation_mtx.columns, cmap='Blues', annot=True)
    title('Filtered Correlation Analysis')
    savefig(f'images/filtered_correlation_analysis_{THRESHOLD}.png')
