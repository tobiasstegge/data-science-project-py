import matplotlib.pyplot as plt
from ds_labs.ds_charts import get_variable_types, bar_chart


def dimensionality(data):
    nr_records = data.shape[0]
    nr_variables = data.shape[1]

    bar_chart(['Nr. of Records', 'Nr. of Variables'], [nr_records, nr_variables], title='Data Dimensionality')
    plt.savefig('./images/dimensionality.png')
    # plt.show()

    # variable types
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    plt.figure(figsize=(4, 2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    plt.savefig('./images/variable_types.png')
    # plt.show()

    # missing values
    missing_values = {}
    for var in data:
        amount_missing = data[var].isna().sum()
        if amount_missing > 0:
            missing_values[var] = amount_missing

    plt.figure(figsize=(8, 8))
    bar_chart(list(missing_values.keys()), list(missing_values.values()), title='Nr of missing values per variable',
              xlabel='variables', ylabel='nr missing values', rotation=True)
    plt.savefig('./images/missing_variables.png')
    # plt.show()


def distribution(data):
    # numeric values
    data_cleaned = data['PERSON_AGE'].dropna().values

    # boxplot
    fig, ax = plt.subplots()
    ax.boxplot(data_cleaned)
    ax.set_title('Boxplot for Person Age')
    ax.set_ylim([0, 150])
    plt.savefig('./images/boxplot_person_age.png')

    # n of outliers for PERSON_AGE
    NR_STDEV = 2
    summary = data.describe()['PERSON_AGE']
    iqr = 1.5 * (summary['75%'] - summary['25%'])  # inter-quartile range with factor
    std = summary['std'] * NR_STDEV

    outliers_age_iqr = data_cleaned[data_cleaned > summary['75%'] + iqr].size + \
                       data_cleaned[data_cleaned < summary['25%'] - iqr].size

    outliers_age_std = data_cleaned[data_cleaned > summary['mean'] + std].size + \
                       data_cleaned[data_cleaned < summary['mean'] - std].size

    outliers = {"iqr": outliers_age_iqr,
                "std": outliers_age_std}

    plt.figure()
    bar_chart(list(outliers.keys()), list(outliers.values()), title="N. of Outliers")
    plt.savefig('./images/n_of_outliers.png')

    # symbolic values
    symbolic_vars = get_variable_types(data)['Symbolic']
    symbolic_vars.remove('PED_ACTION')
    symbolic_vars.remove('PERSON_ID')
    symbolic_vars.remove('CONTRIBUTING_FACTOR_2')

    for idx, var in enumerate(symbolic_vars):
        plt.figure()
        values_counts = data[var].value_counts()
        bar_chart(values_counts.keys(), values_counts.values, title=f'Histogram for {var}', rotation=True)
        plt.savefig(f'./images/histograms_symbolic_{var}.png')
