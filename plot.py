import matplotlib.pyplot as plt


def plot(data, column):
    plt.figure()
    plt.plot(data[column])


def stem(data, column):
    fig, ax = plt.subplots()
    ax.stem(data[column].keys(), data[column])
    plt.show()
