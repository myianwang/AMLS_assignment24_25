# SVM Model
from medmnist import BreastMNIST


def load_data():
    """
    Load the BreastMNIST dataset and preprocess the data.
    :return x_train: training images
    :return y_train: training labels
    :return x_val: validation images
    :return y_val: validation labels
    :return x_test: test images
    :return y_test: test labels
    """
    train_data = BreastMNIST(split='train', download=True, size=28)
    val_data = BreastMNIST(split='val', download=True, size=28)
    test_data = BreastMNIST(split='test', download=True, size=28)

    x_train, y_train = train_data.imgs, train_data.labels.ravel()
    x_val, y_val = val_data.imgs, val_data.labels.ravel()
    x_test, y_test = test_data.imgs, test_data.labels.ravel()

    return x_train, y_train, x_val, y_val, x_test, y_test
