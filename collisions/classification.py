from sklearn.model_selection import train_test_split
from numpy import delete, argwhere
from ds_labs.ds_charts import multiple_bar_chart
from matplotlib.pyplot import figure, savefig


def split_data(data, target, positive, negative, train_size):
    print(f'Splitting for train & test data using hold-out for targer {target}')
    values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

    values_target = data.pop(target).values
    values_other = data.values

    train_other, test_other, train_target, test_target = train_test_split(values_other, values_target,
                                                                          train_size=train_size, stratify=values_target)

    # display for graph
    values['Train'] = [len(delete(train_target, argwhere(train_target == negative))),
                       len(delete(train_target, argwhere(train_target == positive)))]
    values['Test'] = [len(delete(test_target, argwhere(test_target == negative))),
                      len(delete(test_target, argwhere(test_target == positive)))]

    figure(figsize=(12, 4))
    multiple_bar_chart([positive, negative], values, title='Data distribution per dataset')
    savefig('images/split_train_test.png')

    return train_other, test_other, train_target, test_target
