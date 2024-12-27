import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

    # Load BreastMNIST training, validation, and test datasets
    train_data = BreastMNIST(split='train', download=True, size=28)
    val_data = BreastMNIST(split='val', download=True, size=28)
    test_data = BreastMNIST(split='test', download=True, size=28)

    # Get the images and labels
    x_train, y_train = train_data.imgs, train_data.labels
    x_val, y_val = val_data.imgs, val_data.labels
    x_test, y_test = test_data.imgs, test_data.labels

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model():
    """
    Build a Convolutional Neural Network (CNN) model.
    :return model: CNN model
    """

    # Define the CNN model
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, x_val, y_val):
    """
    Train the CNN model on the training data.
    :param model: CNN model
    :param x_train: training images
    :param y_train: training labels
    :param x_val: validation images
    :param y_val: validation labels
    :return history: training history
    """

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32
    )

    return history


def plot_training_history(history):
    """
    Plot the training history of the CNN model.
    :param history: training history
    """

    # Plot the training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Task A (CNN): Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix for CNN model.
    :param y_true: True labels
    :param y_pred: Predicted labels
    """

    # Plot confusion matrix for CNN
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot()
    plt.title("Task A (CNN): Confusion Matrix")
    plt.show()


def main():
    """
    Main function to load data, build, train and evaluate the CNN model.
    """

    # Load the data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    # Build and compile CNN model
    model = build_model()

    # Train the model
    history = train_model(model, x_train, y_train, x_val, y_val)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')

    # Plot training history
    plot_training_history(history)

    # Plot confusion matrix
    plot_confusion_matrix(y_test.argmax(axis=1), model.predict(x_test).argmax(axis=1))


if __name__ == "__main__":
    main()
