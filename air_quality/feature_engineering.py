def feature_selection(data, threshold_correlation, threshold_variance):
    # dropping redundant columns
    correlation_mtx = abs(data.corr())
    vars_2_drop = {}
    for column in correlation_mtx.columns:
        columns_corr = correlation_mtx[column].loc[correlation_mtx[column] >= threshold_correlation]
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
    dropped_correlations = data.drop(columns=selected_2_drop)

    # variance
    var_2_drop_variance = []
    var_2_drop_variances = []
    variances_columns = []
    variances_values = []
    variable_types = get_variable_types(dropped_correlations)
    numeric_vars = variable_types['Numeric']
    data_manual_num = dropped_correlations[numeric_vars]
    for column in data_manual_num.columns:
        value = dropped_correlations[column].var()
        variances_columns.append(column)
        variances_values.append(value)
        if value <= threshold_variance:
            var_2_drop_variance.append(column)
            var_2_drop_variances.append(value)
    figure(figsize=[14, 7])
    bar_chart(variances_columns, variances_values, title='Variance analysis', xlabel='variables', ylabel='variance', rotation=True)
    savefig('images/filtered_variance_analysis.png')
    print(f'Found {len(var_2_drop_variance)} variables with variance under {threshold_variance} dropped: {var_2_drop_variance}')
    return dropped_correlations.drop(columns=var_2_drop_variance)
