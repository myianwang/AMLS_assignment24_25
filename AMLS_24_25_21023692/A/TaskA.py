# Python Script for Task A
# Import the data from the medmnist for task A
from medmnist import BreastMNIST

train_data = BreastMNIST(split='train', download=True, size=64)
val_data = BreastMNIST(split='val', download=True, size=64)
test_data = BreastMNIST(split='test', download=True, size=64)
print(test_data)
print(val_data)
print(train_data)

# CNN Model can be used for Task A
