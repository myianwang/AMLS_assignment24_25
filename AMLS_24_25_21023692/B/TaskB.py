# Python Script for Task B
# Import the data from the medmnist for task B
from medmnist import ChestMNIST
train_data = ChestMNIST(split='train', download=True, size=64)
val_data = ChestMNIST(split='val', download=True, size=64)
test_data = ChestMNIST(split='test', download=True, size=64)
print(test_data)
print(val_data)
print(train_data)
