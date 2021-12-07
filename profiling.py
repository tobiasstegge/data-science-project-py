import matplotlib.pyplot as plt


def show_dimensionality(data):
    nr_records = data.shape[0]
    nr_variables = data.shape[1]

    fig, ax = plt.subplots()
    values = [nr_records, nr_variables]
    bar_plot = plt.bar(['Nr. of Records', 'Nr. of Variables'], [nr_records, nr_variables])

    for idx, rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                values[idx],
                ha='center', va='bottom', rotation=0)

    max_value = max([nr_variables + nr_records])
    plt.ylim(0, max_value + max_value * 0.15)
    plt.title('Data Dimensionality')
    plt.show()


