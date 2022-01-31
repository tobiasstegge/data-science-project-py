from sklearn.naive_bayes import GaussianNB
from matplotlib.pyplot import savefig
from pandas import unique
from ds_labs.ds_charts import plot_evaluation_results


def naive_bayes(train_other, test_other, train_target, test_target, balancing_type):
    model = GaussianNB()
    model.fit(train_other, train_target)
    prediction_train = model.predict(train_other)
    prediction_test = model.predict(test_other)

    labels = unique(train_target).sort_values()
    plot_evaluation_results(labels, train_target, prediction_train, test_target, prediction_test)
    savefig(f'images/classification/nb_best_{str.lower(balancing_type)}.png')
    return model
