from matplotlib.pyplot import subplots, savefig, figure, title, bar, xticks
from ds_labs.ds_charts import get_variable_types, bar_chart, choose_grid, HEIGHT, multiple_bar_chart
from seaborn import heatmap
from collections import Counter


def dimensionality(data):
    nr_records = data.shape[0]
    nr_variables = data.shape[1]

    bar_chart(['Nr. of Records', 'Nr. of Variables'], [nr_records, nr_variables], title='Data Dimensionality')
    savefig('./images/profiling/dimensionality.png')
    # plt.show()

    # variable types
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4, 2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    savefig('./images/profiling/variable_types.png')

    # missing values
    missing_values = {}
    for var in data:
        amount_missing = data[var].isna().sum()
        if amount_missing > 0:
            missing_values[var] = amount_missing

    figure(figsize=(8, 8))
    bar_chart(list(missing_values.keys()), list(missing_values.values()), title='Nr of missing values per variable',
              xlabel='variables', ylabel='nr missing values', rotation=True)
    savefig('./images/profiling/missing_variables.png')


def distribution(data):
    # numeric values
    numeric_data = get_variable_types(data)['Numeric']

    # remove NaN values
    data_cleaned = {}
    for key in numeric_data:
        data_cleaned[key] = data[key].dropna().values

    # boxplot
    print('Making boxplots')
    numeric_vars = get_variable_types(data)['Numeric']
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    i, j = 0, 0
    for idx, var in enumerate(numeric_vars):
        axs[i, j].set_title('Boxplot for %s' % var)
        axs[i, j].boxplot(data[var].dropna().values)
        i, j = (i + 1, 0) if (idx + 1) % cols == 0 else (i, j + 1)
    savefig('images/profiling/single_boxplots.png')

    # n of outliers
    print('Making Outlier Diagrans')
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
    savefig('images/profiling/outliers.png')

    # histograms numeric data
    print('Making Histograms of Numeric Data')
    numeric_vars = get_variable_types(data)['Numeric']
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s' % numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(data[numeric_vars[n]].dropna().values, bins=100)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig('images/profiling/histograms_numeric.png')

    # symbolic values
    print('Making Histogram of Symbolic Data')
    symbolic_vars = get_variable_types(data)['Symbolic']
    for idx, var in enumerate(symbolic_vars):
        figure(figsize=(10, 5))
        occurrences_counts = data[var].value_counts()
        value_counts = Counter(occurrences_counts)
        bar_chart([str(i) for i in  list(value_counts.keys())], list(value_counts.values()), title=f'Chart of Number of Occurrences of Cities for {var}')
        savefig(f'./images/profiling/histograms_symbolic/barchart_occurrences_{var}.png')
        for idx, occurrences_split in enumerate([occurrences_counts[i:i + 100] for i in range(0, len(occurrences_counts), 100)]):
            figure(figsize=(40, 25))
            bar_chart([str(key) for key in occurrences_split.keys()], occurrences_split.values, title=f'Histogram for {var}',
                      rotation=True)
            savefig(f'./images/profiling/histograms_symbolic/histograms_symbolic_{var}_{idx + 1}.png')


def sparsity(data):
        # scatter
        print('Creating Scatterplots')
        numeric_vars = get_variable_types(data)['Numeric']
        numeric_vars_means = [var for var in numeric_vars if var[-4:] == 'Mean']

        # print scatter plots means
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
        savefig(f'images/profiling/sparsity_study_numeric_means.png')

        # scatter plots for all
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
        savefig(f'images/profiling/sparsity_study_numeric.png')

        # heatmap for means
        data_numeric_means = data[numeric_vars_means]
        print("Creating Heatmap for Means")
        figure(figsize=[16, 16])
        corr_mtx = abs(data_numeric_means.corr())
        heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
        title('Correlation analysis for just Means of Values')
        savefig(f'images/profiling/heatmap_means.png')

        # heatmap for all
        data_numeric = data[numeric_vars]
        print("Creating Heatmap for All Numeric")
        figure(figsize=[16, 16])
        corr_mtx = abs(data_numeric.corr())
        heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
        title('Correlation analysis for all Values')
        savefig(f'images/profiling/heatmap.png')

