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


def load_data(train_x_location, train_y_location, test_x_location):
    converter = {0: category_str_to_int}
    train_x = np.loadtxt(train_x_location, delimiter=',', converters=converter)
    train_y = np.loadtxt(train_y_location)
    test_x = np.loadtxt(test_x_location, delimiter=',', converters=converter)
    return train_x, train_y, test_x


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
    # p = np.random.permutation(x.shape[0])
    # return x[p], y[p]
    seed = 23
    s = np.arange(x.shape[0])
    np.random.seed(seed)
    np.random.shuffle(s)
    return x[s], y[s]


def normalize_data(x, test_x):
    # min-max normalization
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    x = (x - min_val) / (max_val - min_val)
    test_x = (test_x - min_val) / (max_val - min_val)
    return x, test_x


def pa_train(train_x, train_y):
    epochs = 50
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
    return w


def svm_train(train_x, train_y):
    regularization = 0.1
    epochs = 50
    d = train_x.shape[1]
    eta = 0.01
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
    return w


def perceptron_train(train_x, train_y):
    epochs = 30
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
    return w


def predict_all_algos(x, w_perceptron, w_svm, w_pa):
    for t in range(x.shape[0]):
        predicted_class_perceptron = np.argmax(np.dot(w_perceptron, x[t]))
        predicted_class_svm = np.argmax(np.dot(w_svm, x[t]))
        predicted_class_pa = np.argmax(np.dot(w_pa, x[t]))
        print('{0}: {1}, {2}: {3}, {4}: {5}'
              .format('perceptron', predicted_class_perceptron, 'svm', predicted_class_svm, 'pa', predicted_class_pa))


def main():
    train_x, train_y, test_x = load_data(sys.argv[1], sys.argv[2], sys.argv[3])

    train_x = one_hot_encode(train_x, 0)
    test_x = one_hot_encode(test_x, 0)
    train_x, test_x = normalize_data(train_x, test_x)

    w_perceptron = perceptron_train(train_x, train_y)
    w_svm = svm_train(train_x, train_y)
    w_pa = pa_train(train_x, train_y)

    predict_all_algos(test_x, w_perceptron, w_svm, w_pa)


if __name__ == "__main__":
    main()
