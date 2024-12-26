from medmnist import BreastMNIST
from sklearn.preprocessing import StandardScaler


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


def preprocess_data(x_train, x_val, x_test):
    """
    Preprocess the data by reshaping the images and standardizing the data.
    :param x_train: training images
    :param x_val: validation images
    :param x_test: test images
    :return x_train_prepared: prepared training images
    :return x_val_prepared: prepared validation images
    :return x_test_prepared: prepared test images
    """

    # Reshape the images
    x_train_prepared = x_train.reshape(len(x_train), -1)
    x_val_prepared = x_val.reshape(len(x_val), -1)
    x_test_prepared = x_test.reshape(len(x_test), -1)

    # Standardize the data
    scaler = StandardScaler()
    x_train_prepared = scaler.fit_transform(x_train_prepared)
    x_val_prepared = scaler.transform(x_val_prepared)
    x_test_prepared = scaler.transform(x_test_prepared)

    return x_train_prepared, x_val_prepared, x_test_prepared
