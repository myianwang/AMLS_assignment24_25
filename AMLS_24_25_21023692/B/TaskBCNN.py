import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from medmnist import BloodMNIST


def load_data():
    """
    Load the BloodMNIST dataset and preprocess the data.
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
    x_train, y_train = train_data.imgs, train_data.labels
    x_val, y_val = val_data.imgs, val_data.labels
    x_test, y_test = test_data.imgs, test_data.labels

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=8)
    y_val = to_categorical(y_val, num_classes=8)
    y_test = to_categorical(y_test, num_classes=8)

    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model():
    """
    Build a Convolutional Neural Network (CNN) model.
    :return model: CNN model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Train the model
def train_model(model, x_train, y_train, x_val, y_val):
    """
    Train the CNN model.
    :param model: CNN model
    :param x_train: training images
    :param y_train: training labels
    :param x_val: validation images
    :param y_val: validation labels
    :returnhistory: training history
    """

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping]
    )
    return history


# Plot training history
def plot_training_history(history):
    """
    Plot the training history of the CNN model.
    :param history: training history
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Task B (CNN): Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix for a given set of true and predicted labels.
    :param y_true: True labels
    :param y_pred: Predicted labels
    """

    # Plot confusion matrix for CNN
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(8)]).plot()
    plt.title("Task B (CNN): Confusion Matrix (Normalized)")
    plt.show()
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(8)]).plot()
    plt.title("Task B (CNN): Confusion Matrix")
    plt.show()


# Main function
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

    return test_acc


def average_main(run_number):
    """
    Run main function multiple times and calculate the average accuracy.
    :param run_number: number of times to run the main function
    """

    # List to store accuracy values
    accuracy_list = []

    # Run the main function multiple times
    for i in range(0, run_number):
        accuracy_list.append(main())

    # Print the average accuracy and the list of accuracy values
    print("Average Accuracy:", sum(accuracy_list) / run_number)
    print("Accuracy List:", accuracy_list)


if __name__ == "__main__":
    main()
