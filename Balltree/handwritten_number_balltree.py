import time
import numpy as np
import os
import matplotlib.pyplot as plt

## Load the training set
train_data = np.load('NN_MNIST/train_data.npy')
train_labels = np.load('NN_MNIST/train_labels.npy')

## Load the testing set
test_data = np.load('NN_MNIST/test_data.npy')
test_labels = np.load('NN_MNIST/test_labels.npy')

# define a fruction that takes an index into a particular data set ("train" or "test) and displays that image
def vis_image(index, dataset = "train"):
    if(dataset == "train"):
        show_digit(train_data[index,])
        label = train_labels[index]
    else:
        show_digit(test_data[index,])
        label = test_labels[index]
    print("label" + str(label))
    return
# Define a function that displays a digit given its vector representation
def show_digit(data_set):
    plt.axis('off')
    plt.imshow(data_set.reshape((28,28)), cmap = plt.cm.gray)
    plt.show()
    return
# computer the number of train labels, and test labels
train_digits, train_counts = np.unique(train_labels, return_counts=True)
test_digits, test_count  = np.unique(test_labels, return_counts=True)
# print(train_digits, train_counts)

# see if vis_image works,
"""
vis_image(2, 'train')
vis_image(3,'test')

"""

# Use sklearn embedded method, Balltree
# import time
from sklearn.neighbors import BallTree

# build nearest neighbor structure on training data
t_before = time.time()
ball_tree = BallTree(train_data)
t_after = time.time()

# training time
t_training = t_after - t_before
print("Time to build trainng structure", t_training)

# get the nearest neighbor predictions on testing data
test_before = time.time()
test_neighbors = ball_tree.query(test_data,k=1,return_distance = False)
ball_tree_prediction = train_labels[test_neighbors]
test_after = time.time()

# compute testing time
time_test = test_after - test_before
print(f"Time to classify test {time_test}  (second)")

# Verify the length of "ball_tree_prediction" and if same as test_labels
print(f"The length of ball tree prediciton {len(ball_tree_prediction)}")
print(f"The length of test labels is {len(test_labels)}")

try:
    if len(ball_tree_prediction) == len(test_labels):
        print("The same length")
    j = 0
    for i in range(len(ball_tree_prediction)):
        if test_labels[i] == ball_tree_prediction[i]:
            j = j + 1
    print(f"The accuracy rate is {j/len(ball_tree_prediction)}")
except ValueError:
    print(" The length of prediction is different from the length of test labels")

# compute ball tree accuracy

