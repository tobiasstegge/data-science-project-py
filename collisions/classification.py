from ds_labs.ds_charts import plot_evaluation_results, multiple_line_chart, bar_chart, horizontal_bar_chart, \
    plot_overfitting_study
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, CategoricalNB
from pandas import unique
from matplotlib.pyplot import figure, savefig, subplots, imread, imshow, axis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, f1_score
from subprocess import call
from numpy import argsort, arange, std
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from ds_labs.ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def naive_bayes(train_other, test_other, train_target, test_target):
    # picking best model
    models = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB(),
              'CategoricalNB': CategoricalNB()
              }
    xvalues = []
    yvalues = []
    for model in models:
        xvalues.append(model)
        models[model].fit(train_other.copy(), train_target.copy())
        prediction_target = models[model].predict(test_other)
        yvalues.append(f1_score(test_target, prediction_target, pos_label='Killed'))

    figure(figsize=[5, 5])
    bar_chart(xvalues, yvalues, ylabel='f1-score', percentage=True)
    savefig(f'images/classification/nb/nb_study.png')

    # run model
    model = MultinomialNB()
    model.fit(train_other, train_target)
    prediction_train = model.predict(train_other)
    prediction_test = model.predict(test_other)

    acc_score = accuracy_score(test_target, prediction_test)
    f1 = f1_score(test_target, prediction_test, pos_label="Killed", average="macro")
    print(f'acc {round(acc_score, 4)} & f1 {round(f1, 4)}')

    labels = unique(train_target)
    labels.sort()
    plot_evaluation_results(labels, train_target, prediction_train, test_target, prediction_test)
    savefig('images/classification/nb/nb.png')
    return model


def knn(train_other, train_target, test_other, test_target):
    # Try best parameters for n of neighbors
    labels = unique(train_target)
    labels.sort()

    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0
    for d in dist:
        target_test_values = []
        print(f'KNN modeling for {d} distance')
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d, n_jobs=-1)
            knn.fit(train_other, train_target)
            predict_test_target = knn.predict(test_other)
            target_test_values.append(f1_score(test_target, predict_test_target, pos_label='Killed'))
            if target_test_values[-1] > last_best:
                best = (n, d)
                last_best = target_test_values[-1]
        values[d] = target_test_values

    figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    savefig('images/classification/knn/knn_study.png')
    print('Best results with %d neighbors and %s' % (best[0], best[1]))

    model = KNeighborsClassifier(n_neighbors=best[0], metric=best[1], n_jobs=-1)
    model.fit(train_other, train_target)
    prediction_train = model.predict(train_other)
    prediction_test = model.predict(test_other)

    acc_score = accuracy_score(test_target, prediction_test)
    f1 = f1_score(test_target, prediction_test, pos_label='Killed')
    print(f'acc {round(acc_score, 10)} f1 {f1}')

    plot_evaluation_results(labels, train_target, prediction_train, test_target, prediction_test)
    savefig(f'images/classification/knn/knn.png')
    return model


def decision_trees(train_other, train_target, test_other, test_target, train):
    min_impurity_decrease = [0.0005, 0.0001, 0.00001, 0.000001]
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
                yvalues.append(f1_score(test_target, prdY, pos_label='Killed'))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_model = tree

            values[d] = yvalues
        multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                            xlabel='min_impurity_decrease', ylabel='f1-score', percentage=True)
    savefig(f'images/classification/tree/dt_study.png')
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f' % (
        best[0], best[1], best[2], last_best))

    dot_data = export_graphviz(best_model, out_file='images/classification/tree/best_tree.dot', filled=True,
                               rounded=True,
                               special_characters=True)

    call(['dot', '-Tpng', 'images/classification/tree/best_tree.dot', '-o', 'images/classification/tree/best_tree.png',
          '-Gdpi=600'])
    figure(figsize=(14, 18))
    imshow(imread('images/classification/tree/best_tree.png'))
    axis('off')

    labels = unique(train_target)
    labels = [str(value) for value in labels]
    labels.sort()
    prd_trn = best_model.predict(train_other)
    prd_tst = best_model.predict(test_other)
    plot_evaluation_results(labels, train_target, prd_trn, test_target, prd_tst)
    savefig(f'images/classification/tree/dt_best.png')

    variables = train.columns
    importances = best_model.feature_importances_
    indices = argsort(importances)[::-1]
    elems = []
    imp_values = []
    for f in range(len(variables) - 1):
        elems += [variables[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    figure(figsize=[5, 5])
    horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance',
                         ylabel='variables')
    savefig(f'images/classification/tree/dt_ranking.png')

    imp = 0.0001
    f = 'entropy'
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for d in max_depths:
        tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
        tree.fit(train_other, train_target)
        prdY = tree.predict(train_other)
        prd_tst_Y = tree.predict(test_other)
        prd_trn_Y = tree.predict(train_other)
        y_tst_values.append(eval_metric(test_target, prd_tst_Y))
        y_trn_values.append(eval_metric(train_target, prd_trn_Y))

    figure()
    plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth',
                           ylabel=str(eval_metric))
    savefig(f'images/classification/tree/dt_overfitting_study.png')


def random_forrest(train_other, train_target, test_other, test_target, train):
    n_estimators = [10, 15, 20, 25]
    max_depths = [30, 40, 50]
    max_features = [0.05, 0.1, 0.2 ,.3]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for f in max_features:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, n_jobs=-1)
                rf.fit(train_other, train_target)
                prdY = rf.predict(test_other)
                yvalues.append(f1_score(test_target, prdY, pos_label='Killed'))
                if yvalues[-1] > last_best:
                    best = (d, f, n)
                    last_best = yvalues[-1]
                    best_model = rf

            values[f] = yvalues
        multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                            xlabel='nr estimators', ylabel='accuracy', percentage=True)
    savefig(f'images/classification/forrest/_rf_study.png')
    print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f' % (
        best[0], best[1], best[2], last_best))

    labels = unique(train_target)
    labels.sort()
    prd_trn = best_model.predict(train_other)
    prd_tst = best_model.predict(test_other)
    plot_evaluation_results(labels, train_target, prd_trn, test_target, prd_tst)
    savefig(f'images/classification/forrest/_rf_best.png')

    variables = train.columns
    importances = best_model.feature_importances_
    stdevs = std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    indices = argsort(importances)[::-1]
    elems = []
    for f in range(len(variables) -1):
        elems += [variables[indices[f]]]
        print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance',
                         xlabel='importance', ylabel='variables')
    savefig(f'images/classification/forrest/_rf_ranking.png')


def gradient_boost(train_other, train_target, test_other, test_target, train):
    labels = unique(train_target)
    labels.sort()

    n_estimators = [60 ,65, 70]
    max_depths = [25, 50, 100]
    learning_rate = [0.8, .9, .95, .99]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(max_depths)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in n_estimators:
                gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                gb.fit(train_other, train_target)
                prdY = gb.predict(test_other)
                yvalues.append(f1_score(test_target, prdY, pos_label='Killed'))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = gb
            values[lr] = yvalues
        multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Gradient Boorsting with max_depth={d}',
                            xlabel='nr estimators', ylabel='accuracy', percentage=True)
    savefig(f'images/classification/gradient/gb_study.png')
    print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f' % (
        best[0], best[1], best[2], last_best))

    prd_trn = best_model.predict(train_other)
    prd_tst = best_model.predict(test_other)
    plot_evaluation_results(labels, train_target, prd_trn, test_target, prd_tst)
    savefig(f'images/classification/gradient/_gb_best.png')

    variables = train.columns
    importances = best_model.feature_importances_
    indices = argsort(importances)[::-1]
    stdevs = std([tree[0].feature_importances_ for tree in best_model.estimators_], axis=0)
    elems = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Gradient Boosting Features importance',
                         xlabel='importance', ylabel='variables')
    savefig(f'images/classification/gradient/_gb_ranking.png')

    lr = 0.7
    max_depth = 10
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in n_estimators:
        gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
        gb.fit(train_other, train_target)
        prd_tst_Y = gb.predict(test_other)
        prd_trn_Y = gb.predict(train_other)
        y_tst_values.append(eval_metric(test_target, prd_tst_Y))
        y_trn_values.append(eval_metric(train_target, prd_trn_Y))
    plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'GB_depth={max_depth}_lr={lr}',
                           xlabel='nr_estimators', ylabel=str(eval_metric))


def mlp(train_other, train_target, test_other, test_target, train):
    file_tag = 'classification/mlp/'
    labels = unique(train_target)
    labels.sort()
    lr_type = ['constant', 'invscaling', 'adaptive']
    lr_type = ['constant']
    max_iter = [100, 200, 300, 400]
    learning_rate = [.98, .99, ]
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(lr_type)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(lr_type)):
        d = lr_type[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in max_iter:
                mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                    learning_rate_init=lr, max_iter=n, verbose=False)
                mlp.fit(train_other, train_target)
                prdY = mlp.predict(test_other)
                yvalues.append(f1_score(test_target, prdY, pos_label='Killed'))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_model = mlp
            values[lr] = yvalues
        multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                            xlabel='mx iter', ylabel='accuracy', percentage=True)
    savefig(f'images/{file_tag}_mlp_study.png')
    print(
        f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')

    prd_trn = best_model.predict(train_other)
    prd_tst = best_model.predict(test_other)
    plot_evaluation_results(labels, train_target, prd_trn, test_target, prd_tst)
    savefig(f'images/{file_tag}_mlp_best.png')

    lr_type = 'constant'
    lr = 0.1
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in max_iter:
        mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type, learning_rate_init=lr,
                            max_iter=n, verbose=False)
        mlp.fit(train_other, train_target)
        prd_tst_Y = mlp.predict(test_other)
        prd_trn_Y = mlp.predict(train_other)
        y_tst_values.append(eval_metric(test_target, prd_tst_Y))
        y_trn_values.append(eval_metric(train_target, prd_trn_Y))
    plot_overfitting_study(max_iter, y_trn_values, y_tst_values, name=f'NN_lr_type={lr_type}_lr={lr}',
                           xlabel='nr episodes', ylabel=str(eval_metric))