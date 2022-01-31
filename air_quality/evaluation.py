from sklearn.metrics import plot_roc_curve
from matplotlib.pyplot import figure, savefig, gca, Axes
from re import sub


def roc_chart(model, test_other, test_target, target: str = '', ax: Axes = None):
    print('Creating ROC Chart')
    figure()
    if ax is None:
        ax = gca()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('FP rate')
    ax.set_ylabel('TP rate')
    ax.set_title('ROC chart for %s' % target)

    ax.plot([0, 1], [0, 1], color='navy', label='random', linewidth=1, linestyle='--',  marker='')
    for clf in model.keys():
        plot_roc_curve(model[clf], test_other, test_target, ax=ax, marker='', linewidth=1)
    ax.legend(loc="lower right")
    savefig(f'images/classification/roc_charts/{sub("[^A-Za-z0-9]+","",str.lower(str(model)))}roc_chart.png')