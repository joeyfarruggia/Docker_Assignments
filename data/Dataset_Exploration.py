import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tarfile
import matplotlib.pyplot as plt

#Extracting the dataset
tar = tarfile.open(
    r"C:\Users\Joey\OneDrive\Documents\GitHub\Docker_Assignments\data\cifar-10-python.tar.gz"
)
tar.extractall()
tar.close()

#Define a function to load the batch file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Load dataset batch files
data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')

#Combine the loaded batches into a single dataset
X_train = np.concatenate([
data_batch_1[b'data'], 
data_batch_2[b'data'], 
data_batch_3[b'data'], 
data_batch_4[b'data'], 
data_batch_5[b'data']
])

y_train = np.concatenate([
data_batch_1[b'labels'],
data_batch_2[b'labels'],
data_batch_3[b'labels'],
data_batch_4[b'labels'],
data_batch_5[b'labels']
])

#Load the test batch
test_batch = unpickle('cifar-10-batches-py/test_batch')
X_test = test_batch[b'data']
y_test = np.array(test_batch[b'labels'])

#Reshape the data
X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

#Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Verify the dataset extraction
print("Dataset extracted and loaded successfully.")

#Check the dataset shape
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

# CIFAR-10 category names.
# Each position in this list corresponds to a numeric label from 0 through 9.
class_names = [
    "Airplane",      # Label 0
    "Automobile",    # Label 1
    "Bird",          # Label 2
    "Cat",           # Label 3
    "Deer",          # Label 4
    "Dog",           # Label 5
    "Frog",          # Label 6
    "Horse",         # Label 7
    "Ship",          # Label 8
    "Truck"          # Label 9
]

# Visualize 15 images in 3 rows and 5 columns.
fig, axes = plt.subplots(3, 5, figsize=(12, 9))

for i, ax in enumerate(axes.flat):
    label_number = y_train[i]
    category_name = class_names[label_number]

    ax.imshow(X_train[i])
    ax.set_title(
        f"Label: {label_number}\nCategory: {category_name}",
        fontsize=10
    )
    ax.axis("off")

plt.tight_layout()
plt.show()

#Verify class labels
unique_labels = np.unique(y_train)
print("Unique class labels in the training set:", unique_labels)