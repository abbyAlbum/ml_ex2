import sys
import numpy as np

categorical_dict = {
    'M': 0,
    'F': 1,
    'I': 2
}


def category_str_to_int(category):
    return categorical_dict[chr(category[0])]


def load_data(train_x_location, train_y_location, test_x_location):
    # TODO: convert categorical data to continuous
    converter = {0: category_str_to_int}
    train_x = np.loadtxt(train_x_location, delimiter=',', converters=converter)
    train_y = np.loadtxt(train_y_location)
    return train_x, train_y


def shuffle_data(x, y):
    # every day I'm shuffling
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


def normalize_data(x):
    # z-score normalization
    x = (x - np.mean(x)) / (np.std(x))
    return x

    # min-max normalization
    # x = (x - np.max(x)) / (np.max(x) - np.min(x))
    # return x


def error_calc(w, train_x, train_y):
    errors = 0
    for t in range(train_x.shape[0]):
        y_hat = np.argmax(np.dot(w, train_x[t]))
        if y_hat != train_y[t]:
            errors += 1
    print('precision: {0}'.format(1 - (float(errors) / train_x.shape[0])))


def perceptron_train(train_x, train_y):
    num_of_classes = 3
    epochs = 10
    d = train_x.shape[1]
    eta = 0.01
    w = np.zeros((num_of_classes, d))

    for e in range(epochs):
        train_x, train_y = shuffle_data(train_x, train_y)

        for x, y in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                y = int(y)
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x

    error_calc(w, train_x, train_y)


def main():
    train_x, train_y = load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    train_x = normalize_data(train_x)
    # from sklearn.preprocessing import minmax_scale
    # p = minmax_scale(train_x)
    perceptron_train(train_x, train_y)


if __name__ == "__main__":
    main()
