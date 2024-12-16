# Python Script for Task A
# Import the data from the medmnist for task A
from medmnist import BreastMNIST


# Load BreastMNIST dataset
def load_data():
    train_data = BreastMNIST(split='train', download=True, size=224)
    val_data = BreastMNIST(split='val', download=True, size=224)
    test_data = BreastMNIST(split='test', download=True, size=224)

    x_train, y_train = train_data.imgs, train_data.labels
    x_val, y_val = val_data.imgs, val_data.labels
    x_test, y_test = test_data.imgs, test_data.labels

    print(test_data)
    print(val_data)
    print(train_data)


load_data()

# CNN Model can be used for Task A
