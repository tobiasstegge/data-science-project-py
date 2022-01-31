from sklearn.metrics import plot_roc_curve
from pandas import unique
from matplotlib.pyplot import figure, savefig, gca, Axes
from sklearn.metrics import confusion_matrix
from re import sub


def create_confusion_matrix(model, test_target, test_records):
    print('Creating Confusion Matrix')
    labels = unique(test_records)
    labels.sort()
    predicted_target = model.predict(test_target)
    matrix = confusion_matrix(test_records, predicted_target)
    tn, fp, fn, tp = matrix.ravel()
    precision = tp / (tp + fp)
    accuracy = model.score(test_target, test_records)
    error_rate = 1 -accuracy
    print(f'True: {tn + tp}  \n'
          f'False: {fn + fp} \n'
          f'Ratio: {(tp + tn) / (tn + fp + fn + tp)}')


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
    savefig(f'images/classification/{sub("[^A-Za-z0-9]+","",str.lower(str(model)))}_roc_chart.png')