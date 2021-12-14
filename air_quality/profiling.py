from matplotlib.pyplot import subplots, savefig, figure, title
from ds_labs.ds_charts import get_variable_types, bar_chart, choose_grid, HEIGHT, multiple_bar_chart
from seaborn import heatmap


def dimensionality(data):
    nr_records = data.shape[0]
    nr_variables = data.shape[1]

    bar_chart(['Nr. of Records', 'Nr. of Variables'], [nr_records, nr_variables], title='Data Dimensionality')
    savefig('./images/dimensionality.png')
    # plt.show()

    # variable types
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4, 2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    savefig('./images/variable_types.png')

    # missing values
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
    numeric_data = get_variable_types(data)['Numeric']

    # remove NaN values
    data_cleaned = {}
    for key in numeric_data:
        data_cleaned[key] = data[key].dropna().values

    # boxplot
    numeric_vars = get_variable_types(data)['Numeric']
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    i, j = 0, 0
    for idx, var in enumerate(numeric_vars):
        axs[i, j].set_title('Boxplot for %s' % var)
        axs[i, j].boxplot(data[var].dropna().values)
        i, j = (i + 1, 0) if (idx + 1) % cols == 0 else (i, j + 1)
    savefig('images/single_boxplots.png')

    # n of outliers
    NR_STDEV: int = 2
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    outliers_iqr = []
    outliers_stdev = []
    summary = data.describe(include='number')

    for var in numeric_vars:
        iqr = 1.5 * (summary[var]['75%'] - summary[var]['25%'])
        outliers_iqr += [
            data[data[var] > summary[var]['75%'] + iqr].count()[var] +
            data[data[var] < summary[var]['25%'] - iqr].count()[var]]
        std = NR_STDEV * summary[var]['std']
        outliers_stdev += [
            data[data[var] > summary[var]['mean'] + std].count()[var] +
            data[data[var] < summary[var]['mean'] - std].count()[var]]

    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables',
                       ylabel='nr outliers', percentage=False)
    savefig('images/outliers.png')

    # histograms numeric data
    numeric_vars = get_variable_types(data)['Numeric']
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s' % numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig('images/single_histograms_numeric.png')

    # symbolic values
    symbolic_vars = get_variable_types(data)['Symbolic']
    symbolic_vars.remove('City_EN')
    symbolic_vars.remove('GbCity')
    for idx, var in enumerate(symbolic_vars):
        figure()
        values_counts = data[var].value_counts()
        bar_chart(values_counts.keys(), values_counts.values, title=f'Histogram for {var}', rotation=True)
        savefig(f'./images/histograms_symbolic_{var}.png')


def sparsity(data):
        # scatter
        numeric_vars = get_variable_types(data)['Numeric']
        numeric_vars_means = [var for var in numeric_vars if var[-4:] == 'Mean']
        data_means = data[numeric_vars_means]

        rows, cols = len(numeric_vars_means) - 1, len(numeric_vars_means) - 1
        fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
        for i in range(len(numeric_vars_means)):
            var1 = numeric_vars_means[i]
            for j in range(i + 1, len(numeric_vars_means)):
                var2 = numeric_vars_means[j]
                axs[i, j - 1].set_title("%s x %s" % (var1, var2))
                axs[i, j - 1].set_xlabel(var1)
                axs[i, j - 1].set_ylabel(var2)
                axs[i, j - 1].scatter(data[var1], data[var2])
        savefig(f'images/sparsity_study_numeric.png')

        # heatmap
        print("Creating Heatmap")
        figure(figsize=[16, 16])
        corr_mtx = abs(data_means.corr())
        print(corr_mtx)
        heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
        title('Correlation analysis')
        savefig(f'images/correlation_analysis2.png')

