# Python Script for Task B
# Import the data from the medmnist for task B
from medmnist import BloodMNIST
train_data = BloodMNIST(split='train', download=True, size=64)
val_data = BloodMNIST(split='val', download=True, size=64)
test_data = BloodMNIST(split='test', download=True, size=64)
print(test_data)
print(val_data)
print(train_data)

# CNN Model can be used for Task B