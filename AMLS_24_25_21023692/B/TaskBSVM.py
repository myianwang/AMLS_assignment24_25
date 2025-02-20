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


def train_svm_and_evaluate(x_train, y_train, x_val, y_val):
    """
    Train a Support Vector Machine (SVM) model and evaluate it on the validation set.
    :param x_train: training images
    :param y_train: training labels
    :param x_val: validation images
    :param y_val: validation labels
    :return best_svm: best SVM model
    """

    # Grid Search to find the best Hyperparameters
    param_grid = {
        'C': [0.1, 0.5, 0.8, 1, 2, 3, 4, 5, 7, 10, 20, 30],
        'gamma': ['scale', 0.01, 0.05, 0.1],
        'kernel': ['rbf', 'linear']
    }

    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    # Train the best model with the training set
    best_svm = grid_search.best_estimator_
    best_svm.fit(x_train, y_train)

    # Evaluate the model with validation set
    val_preds = best_svm.predict(x_val)
    val_accuracy = accuracy_score(y_val, val_preds)

    print("\nValidation Accuracy:", val_accuracy)
    print("\nClassification Report:\n", classification_report(y_val, val_preds))

    # Plot confusion matrix for validation set
    cm = confusion_matrix(y_val, val_preds)
    ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(8)]).plot()
    plt.title("Task B (SVM):Validation Confusion Matrix")
    plt.show()

    return best_svm


def evaluate_on_test(best_svm, x_test, y_test):
    """
    Evaluate the best SVM model on the test set.
    :param best_svm: best SVM model
    :param x_test: test images
    :param y_test: test labels
    """

    # Evaluate the model on the test set
    test_preds = best_svm.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_preds)

    print("\nTest Accuracy:", test_accuracy)
    print("\nTest Classification Report:\n", classification_report(y_test, test_preds))

    # Plot confusion matrix for test set
    cm = confusion_matrix(y_test, test_preds)
    ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(8)]).plot()
    plt.title("Task B (SVM):Test Confusion Matrix")
    plt.show()
    cm = confusion_matrix(y_test, test_preds, normalize='true')
    ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(8)]).plot()
    plt.title("Task B (SVM):Test Confusion Matrix (Normalized)")
    plt.show()


def main():
    """
    Main function to load data, preprocess, train and evaluate the SVM model.
    """

    # Load BloodMNIST dataset
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    # Preprocess the data
    x_train_prepared, x_val_prepared, x_test_prepared = preprocess_data(x_train, x_val, x_test)

    # Train SVM and evaluate on the validation set
    best_svm = train_svm_and_evaluate(x_train_prepared, y_train, x_val_prepared, y_val)

    # Evaluate the best model on the test set
    evaluate_on_test(best_svm, x_test_prepared, y_test)


if __name__ == "__main__":
    main()
