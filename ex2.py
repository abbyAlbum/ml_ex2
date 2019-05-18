import sys
import numpy as np

categorical_dict = {
    'M': 0,
    'F': 1,
    'I': 2
}

num_of_classes = 3


def category_str_to_int(category):
    return categorical_dict[chr(category[0])]


def split_to_validation(x, y, train_data_percentage=0.8):
    x, y = shuffle_data(x, y)
    split_index = int(x.shape[0] * train_data_percentage)
    train_x, train_y, validation_x, validation_y = x[:split_index, :], y[:split_index], \
                                                   x[split_index:, :], y[split_index:]
    return train_x, train_y, validation_x, validation_y


def load_data(train_x_location, train_y_location, test_x_location, test_y_location):
    converter = {0: category_str_to_int}
    train_x = np.loadtxt(train_x_location, delimiter=',', converters=converter)
    train_y = np.loadtxt(train_y_location)
    test_x = np.loadtxt(test_x_location, delimiter=',', converters=converter)
    test_y = np.loadtxt(test_y_location)
    return train_x, train_y, test_x, test_y


def add_bias(x):
    x_with_bias = np.ones((x.shape[0], x.shape[1] + 1))
    x_with_bias[:, :-1] = x
    return x_with_bias


def one_hot_encode(data, index):
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


def normalize_data(x, test_x):
    # z-score normalization
    # mean = np.mean(x)
    # std = np.std(x)
    # x = (x - mean) / std
    # test_x = (test_x - mean) / std

    # min-max normalization
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    test_x = (test_x - min_val) / (max_val - min_val)
    return x, test_x


def precision_calc(w, train_x, train_y):
    errors = 0
    for t in range(train_x.shape[0]):
        y_hat = np.argmax(np.dot(w, train_x[t]))
        if y_hat != train_y[t]:
            errors += 1
    precision = 1 - (float(errors) / train_x.shape[0])
    print('train precision: {0}'.format(precision))
    return precision


def pa_train(train_x, train_y):
    epochs = 100
    d = train_x.shape[1]
    w = np.zeros((num_of_classes, d))

    for e in range(epochs):
        train_x, train_y = shuffle_data(train_x, train_y)

        for x, y in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                y = int(y)
                hinge_loss = max(0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x))
                tao = hinge_loss / (2 * np.linalg.norm(x))
                w[y, :] = w[y, :] + tao * x
                w[y_hat, :] = w[y_hat, :] - tao * x

    precision_calc(w, train_x, train_y)
    return w


def svm_train(train_x, train_y):
    regularization = 0.1
    epochs = 100
    d = train_x.shape[1]
    eta = 0.001
    w = np.zeros((num_of_classes, d))

    for e in range(epochs):
        train_x, train_y = shuffle_data(train_x, train_y)

        for x, y in zip(train_x, train_y):
            y_hat = int(np.argmax(np.dot(w, x)))
            y = int(y)
            if y_hat != y:
                w[y, :] = (1 - eta * regularization) * w[y, :] + eta * x
                w[y_hat, :] = (1 - eta * regularization) * w[y_hat, :] - eta * x
                remaining_row_index = w.shape[0] - (y + y_hat)
                w[remaining_row_index, :] = (1 - eta * regularization) * w[remaining_row_index, :]
            else:
                w[y, :] = (1 - eta * regularization) * w[y, :] + eta * x
                for i in range(w.shape[0]):
                    if i != y:
                        w[i, :] = (1 - eta * regularization) * w[i, :]
        eta *= 1 - e / epochs
        # regularization *= 1 - e / epochs
        # eta /= np.sqrt(e + 1)
    precision_calc(w, train_x, train_y)
    return w


def perceptron_train(train_x, train_y):
    epochs = 100
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

        eta *= 1 - e / epochs
        # eta /= np.sqrt(e + 1)
    precision_calc(w, train_x, train_y)
    return w


def predict(w, x, y):
    errors = 0
    for t in range(x.shape[0]):
        y_hat = np.argmax(np.dot(w, x[t]))
        if y_hat != y[t]:
            errors += 1
    print('test precision: {0}'.format(1 - (float(errors) / x.shape[0])))


def main():
    train_x, train_y, test_x, test_y = load_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    train_x = one_hot_encode(train_x, 0)
    test_x = one_hot_encode(test_x, 0)
    train_x, test_x = normalize_data(train_x, test_x)
    # train_x = add_bias(train_x)
    # test_x = add_bias(test_x)
    train_x, train_y, validation_x, validation_y = split_to_validation(train_x, train_y)
    # test_x = normalize_data(test_x)
    print('perceptron:')
    w = perceptron_train(train_x, train_y)
    predict(w, validation_x, validation_y)
    predict(w, test_x, test_y)
    print('svm:')
    w = svm_train(train_x, train_y)
    predict(w, validation_x, validation_y)
    predict(w, test_x, test_y)
    print('pa:')
    w = pa_train(train_x, train_y)
    predict(w, validation_x, validation_y)
    predict(w, test_x, test_y)


if __name__ == "__main__":
    main()
