# Python Script for Task B
# Import the data from the medmnist for task B
from medmnist import BloodMNIST


# Load BloodMNIST dataset
def load_data():
    train_data = BloodMNIST(split='train', download=True, size=224)
    val_data = BloodMNIST(split='val', download=True, size=224)
    test_data = BloodMNIST(split='test', download=True, size=224)

    x_train, y_train = train_data.imgs, train_data.labels
    x_val, y_val = val_data.imgs, val_data.labels
    x_test, y_test = test_data.imgs, test_data.labels

    print(test_data)
    print(val_data)
    print(train_data)


load_data()

# CNN Model can be used for Task B
