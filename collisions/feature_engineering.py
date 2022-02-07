from ds_labs.ds_charts import get_variable_types, bar_chart
from seaborn import heatmap
from matplotlib.pyplot import title, savefig, figure


def select_redundant(data, threshold_correlation):
    all_data = data
    correlation_mtx = abs(data.corr())
    figure(figsize=[15, 15])
    heatmap(correlation_mtx, xticklabels=correlation_mtx.columns, yticklabels=correlation_mtx.columns, annot=True,
            cmap='Blues')
    title('Filtered Correlation Analysis')
    savefig(f'images/classification/filtered_correlation_analysis.png')
    vars_2_drop = {}
    for column in correlation_mtx.columns:
        columns_corr = correlation_mtx[column].loc[correlation_mtx[column] >= threshold_correlation]
        if column not in vars_2_drop and (len(columns_corr) > 1):
            vars_2_drop[column] = columns_corr.index.values

    selected_to_drop = []
    print(vars_2_drop.keys())
    for key in vars_2_drop.keys():
        if key not in selected_to_drop:
            for var in vars_2_drop[key]:
                if var != key and var not in selected_to_drop:
                    selected_to_drop.append(var)
    print(f'Variables selected to drop {selected_to_drop} on threshold {threshold_correlation}')

    return all_data.drop(columns=selected_to_drop)


def select_low_variance(data, threshold_variance):
    selected_to_drop = []
    var_2_drop_variances = []
    variances_columns = []
    variances_values = []
    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    data_numeric = data[numeric_vars]
    for column in data_numeric.columns:
        value = data_numeric[column].var()
        variances_columns.append(column)
        variances_values.append(value)
        if value <= threshold_variance:
            selected_to_drop.append(column)
            var_2_drop_variances.append(value)
    print(
        f'Found {len(selected_to_drop)} variables with variance under {threshold_variance} dropped: {selected_to_drop}')
    figure(figsize=[10, 6])
    bar_chart(variances_columns, variances_values, title='Variance analysis', xlabel='variables', ylabel='variance',
              rotation=True)
    savefig('images/classification/filtered_variance_analysis.png')
    return data.drop(columns=selected_to_drop)
