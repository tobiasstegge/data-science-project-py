from ds_labs.ds_charts import plot_evaluation_results, multiple_line_chart
from sklearn.naive_bayes import GaussianNB
from pandas import unique
from matplotlib.pyplot import figure, savefig, subplots, imread, imshow, axis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from subprocess import call


def naive_bayes(train_other, test_other, train_target, test_target):
    model = GaussianNB()
    model.fit(train_other, train_target)
    prediction_train = model.predict(train_other)
    prediction_test = model.predict(test_other)

    labels = unique(train_target).sort_values()
    plot_evaluation_results(labels, train_target, prediction_train, test_target, prediction_test)
    savefig('images/classification/nb_best.png')
    return model


def knn(train_other, train_target, test_other, test_target):
    # Try best parameters for n of neighbors
    labels = unique(train_target).sort_values()

    eval_metric = accuracy_score
    #nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    nvalues = [5]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        target_test_values = []
        for n in nvalues:
            print(f'KNN modeling for n={n} nearest neighbors with {d}')
            knn = KNeighborsClassifier(n_neighbors=n, metric=d, n_jobs=-1)
            knn.fit(train_other, train_target)
            predict_test_target = knn.predict(test_other)
            target_test_values.append(eval_metric(test_target, predict_test_target))
            if target_test_values[-1] > last_best:
                best = (n, d)
                last_best = target_test_values[-1]
        values[d] = target_test_values

    figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    savefig('images/classification/knn_study.png')
    print('Best results with %d neighbors and %s' % (best[0], best[1]))

    model = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    model.fit(train_other, train_target)
    prediction_train = model.predict(train_other)
    prediction_test = model.predict(test_other)
    plot_evaluation_results(labels, train_target, prediction_train, test_target, prediction_test)
    savefig('images/classification/knn_best.png')

    return model


def decision_trees(train_other, train_target, test_other, test_target):
    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    best = ('', 0, 0.0)
    last_best = 0
    best_model = None

    figure()
    fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(train_other, train_target)
                prdY = tree.predict(test_other)
                yvalues.append(accuracy_score(test_target, prdY))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues
        multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                            xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
    savefig(f'images/classification/dt_study.png')
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f' % (
        best[0], best[1], best[2], last_best))

    dot_data = export_graphviz(best_model, out_file='images/best_tree.dot', filled=True, rounded=True,
                               special_characters=True)

    call(['dot', '-Tpng', 'images/best_tree.dot', '-o', 'images/best_tree.png', '-Gdpi=600'])
    figure(figsize=(14, 18))
    imshow(imread('images/best_tree.png'))
    axis('off')

    labels = unique(train_target)
    labels = [str(value) for value in labels]
    prd_trn = best_model.predict(train_other)
    prd_tst = best_model.predict(test_other)
    plot_evaluation_results(labels, train_target, prd_trn, test_target, prd_tst)
    savefig(f'images/classification/dt_best.png')
