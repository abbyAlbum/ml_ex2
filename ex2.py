import sys
import numpy as np

categorical_dict = {
    'M': 0,
    'F': 1,
    'I': 2
}


def category_str_to_int(category):
    return categorical_dict[chr(category[0])]


def load_data(train_x_location, train_y_location, test_x_location, test_y_location):
    converter = {0: category_str_to_int}
    train_x = np.loadtxt(train_x_location, delimiter=',', converters=converter)
    train_y = np.loadtxt(train_y_location)
    test_x = np.loadtxt(test_x_location, delimiter=',', converters=converter)
    test_y = np.loadtxt(test_y_location)
    return train_x, train_y, test_x, test_y


def one_hot_encode(data, index, num_of_classes):
    one_hot_encoded_data = []
    for row in data:
        binary_encoding = [0] * num_of_classes
        category = int(row[index])
        binary_encoding[category] = 1
        row = np.delete(row, 0)
        row = np.append(row, binary_encoding)
        one_hot_encoded_data.append(row)
    return np.asarray(one_hot_encoded_data)


def shuffle_data(x, y):
    # every day I'm shuffling
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


def normalize_data(x):
    # z-score normalization
    x = (x - np.mean(x)) / (np.std(x))
    # min-max normalization
    # x = (x - np.max(x)) / (np.max(x) - np.min(x))
    return x


def precision_calc(w, train_x, train_y):
    errors = 0
    for t in range(train_x.shape[0]):
        y_hat = np.argmax(np.dot(w, train_x[t]))
        if y_hat != train_y[t]:
            errors += 1
    print('precision: {0}'.format(1 - (float(errors) / train_x.shape[0])))


def perceptron_train(train_x, train_y):
    num_of_classes = 3
    epochs = 100000
    d = train_x.shape[1]
    eta = 0.001
    w = np.zeros((num_of_classes, d))

    for e in range(epochs):
        train_x, train_y = shuffle_data(train_x, train_y)

        for x, y in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                y = int(y)
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x

    precision_calc(w, train_x, train_y)
    return w


def perceptron_predict(w, x, y):
    errors = 0
    for t in range(x.shape[0]):
        y_hat = np.argmax(np.dot(w, x[t]))
        if y_hat != y[t]:
            errors += 1
    print('precision: {0}'.format(1 - (float(errors) / x.shape[0])))


def main():
    train_x, train_y, test_x, test_y = load_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    train_x = one_hot_encode(train_x, 0, 3)
    train_x = normalize_data(train_x)
    test_x = one_hot_encode(test_x, 0, 3)
    test_x = normalize_data(test_x)
    # from sklearn.preprocessing import minmax_scale
    # p = minmax_scale(train_x)
    w = perceptron_train(train_x, train_y)
    perceptron_predict(w, test_x, test_y)


if __name__ == "__main__":
    main()
