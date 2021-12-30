from matplotlib.pyplot import subplots, savefig, figure, title
from ds_labs.ds_charts import get_variable_types, bar_chart, HEIGHT, choose_grid
from seaborn import heatmap
from numpy import logical_and
import ds_labs.config as cfg


def dimensionality(data):
    # data dimension
    print("Creating Data Dimension Charts")
    nr_records = data.shape[0]
    nr_variables = data.shape[1]

    bar_chart(['Nr. of Records', 'Nr. of Variables'], [nr_records, nr_variables], title='Data Dimensionality')
    savefig('./images/dimensionality.png')

    # variable types
    print("Creating Variable Types Charts")
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4, 2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    savefig('./images/variable_types.png')

    # missing values
    print("Creating Missing Values Charts")
    missing_values = {}
    for var in data:
        amount_missing = data[var].isna().sum()
        if amount_missing > 0:
            missing_values[var] = amount_missing

    figure(figsize=(8, 8))
    bar_chart(list(missing_values.keys()), list(missing_values.values()), title='Nr of missing values per variable',
              xlabel='variables', ylabel='nr missing values', rotation=True)
    savefig('./images/missing_variables.png')


def distribution(data):
    # numeric values
    data_cleaned = data['PERSON_AGE'].dropna().values

    # boxplot
    fig, ax = subplots()
    ax.boxplot(data_cleaned)
    ax.set_title('Boxplot for Person Age')
    ax.set_ylim([0, 150])
    savefig('./images/boxplot_person_age.png')

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

    figure()
    bar_chart(list(outliers.keys()), list(outliers.values()), title="N. of Outliers", xlabel='PERSON_AGE')
    savefig('./images/n_of_outliers.png')

    # histograms symbolic values
    print("Creating Histograms for Symbolic Variables")
    symbolic_vars = get_variable_types(data)['Symbolic']
    symbolic_vars.remove('PERSON_ID')
    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT * 2, rows * HEIGHT * 2), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = data[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s' % symbolic_vars[n],
                  xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation=False)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig('images/histograms_symbolic.png')

    # histogram binary values
    print("Creating Histograms for Binary Variables")
    fig, axs = subplots()
    counts = data['PERSON_INJURY'].value_counts()
    bar_chart(counts.index.to_list(), counts.values, title='Histogram for PERSON_INJURY',
              xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
    savefig('images/histograms_binary.png')

    # histogram for numeric values
    print('Creating Histograms for Numeric Variables')
    column_age = data['PERSON_AGE']
    bins = 121
    column_age.dropna().values
    mask = logical_and(column_age > 0, column_age < 123)
    column_age_cleaned = column_age[mask]
    fig, axs = subplots()
    axs.set_title(f'Histogram for PERSON_AGE for {bins} bins')
    axs.set_ylabel("nr records")
    axs.hist(column_age_cleaned, bins=bins, color=cfg.LINE_COLOR)
    savefig('images/single_histograms_numeric.png')


def sparsity(data):
    # scatter
    print("Creating Sparsity Plots")
    numeric_vars = get_variable_types(data)['Numeric']
    data_means = data[numeric_vars]

    rows, cols = len(numeric_vars) - 1, len(numeric_vars) - 1
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i + 1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j - 1].set_title("%s x %s" % (var1, var2))
            axs[i, j - 1].set_xlabel(var1)
            axs[i, j - 1].set_ylabel(var2)
            axs[i, j - 1].scatter(data[var1], data[var2])
    savefig(f'images/sparsity_study_numeric.png')

    # heatmap
    print("Creating Heatmap")
    figure(figsize=[16, 16])
    corr_mtx = abs(data_means.corr())
    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation analysis')
    savefig(f'images/correlation_analysis.png')