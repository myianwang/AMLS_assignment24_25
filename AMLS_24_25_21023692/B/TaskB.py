# Python Script for Task B
# Import the data from the medmnist for task B
import medmnist
from medmnist import INFO


# Load BreastMNIST dataset
def load_data():
    info = INFO["bloodmnist"]
    DataClass = getattr(medmnist, info['python_class'])

    train_data = DataClass(split='train', download=True)
    val_data = DataClass(split='val', download=True)
    test_data = DataClass(split='test', download=True)

    print(test_data)
    print(val_data)
    print(train_data)

# CNN Model can be used for Task B
