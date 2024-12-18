# Python Script for Task A
# Import the data from the medmnist for task A
from medmnist import BreastMNIST
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Load BreastMNIST dataset
def load_data():
    train_data = BreastMNIST(split='train', download=True, size=224)
    val_data = BreastMNIST(split='val', download=True, size=224)
    test_data = BreastMNIST(split='test', download=True, size=224)

    x_train, y_train = train_data.imgs, train_data.labels
    x_val, y_val = val_data.imgs, val_data.labels
    x_test, y_test = test_data.imgs, test_data.labels

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    print(test_data)
    print(val_data)
    print(train_data)

    return (x_train, y_train), (x_test, y_test)


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


load_data()

# CNN Model can be used for Task A
