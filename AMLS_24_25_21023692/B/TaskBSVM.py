import matplotlib.pyplot as plt
from medmnist import BloodMNIST
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay


def load_data():
    """
    Load the BloodMNIST dataset.
    :return x_train: training images
    :return y_train: training labels
    :return x_val: validation images
    :return y_val: validation labels
    :return x_test: test images
    :return y_test: test labels
    """

    # Load BloodMNIST training, validation, and test datasets
    train_data = BloodMNIST(split='train', download=True, size=28)
    val_data = BloodMNIST(split='val', download=True, size=28)
    test_data = BloodMNIST(split='test', download=True, size=28)

    # Get the images and labels
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

    # Reshaping the images
    x_train_prepared = x_train.reshape(len(x_train), -1)
    x_val_prepared = x_val.reshape(len(x_val), -1)
    x_test_prepared = x_test.reshape(len(x_test), -1)

    # Standardize the data
    scaler = StandardScaler()
    x_train_prepared = scaler.fit_transform(x_train_prepared)
    x_val_prepared = scaler.transform(x_val_prepared)
    x_test_prepared = scaler.transform(x_test_prepared)

    return x_train_prepared, x_val_prepared, x_test_prepared