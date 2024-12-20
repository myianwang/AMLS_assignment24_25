import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from medmnist import BreastMNIST


# Load BreastMNIST dataset
def load_data():
    # Load BreastMNIST training, validation, and test datasets
    train_data = BreastMNIST(split='train', download=True, size=224)
    val_data = BreastMNIST(split='val', download=True, size=224)
    test_data = BreastMNIST(split='test', download=True, size=224)

    # Get the images and labels
    x_train, y_train = train_data.imgs, train_data.labels
    x_val, y_val = val_data.imgs, val_data.labels
    x_test, y_test = test_data.imgs, test_data.labels

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    return x_train, y_train, x_val, y_val, x_test, y_test


# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(224, 224, 1)),
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

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Train the model
def train_model(model, x_train, y_train, x_val, y_val):
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32
    )
    return history


# Plot training history
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


def main():
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


if __name__ == "__main__":
    main()
