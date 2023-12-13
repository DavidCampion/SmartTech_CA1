import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from keras.datasets import cifar10, cifar100

def load_data():
    # Load CIFAR-10 and CIFAR-100
    (X_train_10, y_train_10), (X_test_10, y_test_10) = cifar10.load_data()
    (X_train_100, y_train_100), (X_test_100, y_test_100) = cifar100.load_data(label_mode='fine')

    print("Original CIFAR-10 training data shape:", X_train_10.shape)
    print("Original CIFAR-100 training data shape:", X_train_100.shape)

    # CIFAR-10 classes
    cifar10_classes = [1, 2, 3, 4, 5, 7, 9]
    # CIFAR-100 classes
    cifar100_classes = [2, 8, 11, 13, 17, 19, 34, 35, 41, 46, 48, 58, 65, 80, 89, 90, 98]

    # Filter CIFAR-10 data
    X_train_10, y_train_10 = X_train_10[np.isin(y_train_10.flatten(), cifar10_classes)], y_train_10[np.isin(y_train_10.flatten(), cifar10_classes)]
    X_test_10, y_test_10 = X_test_10[np.isin(y_test_10.flatten(), cifar10_classes)], y_test_10[np.isin(y_test_10.flatten(), cifar10_classes)]

    print("CIFAR-10 training data shape:", X_train_10.shape)

    # Filter CIFAR-100 data
    X_train_100, y_train_100 = X_train_100[np.isin(y_train_100.flatten(), cifar100_classes)], y_train_100[np.isin(y_train_100.flatten(), cifar100_classes)]
    X_test_100, y_test_100 = X_test_100[np.isin(y_test_100.flatten(), cifar100_classes)], y_test_100[np.isin(y_test_100.flatten(), cifar100_classes)]

    print("CIFAR-100 training data shape:", X_train_100.shape)

    # Combine the datasets
    X_train_combined = np.concatenate((X_train_10, X_train_100), axis=0)
    y_train_combined = np.concatenate((y_train_10, y_train_100), axis=0)
    X_test_combined = np.concatenate((X_test_10, X_test_100), axis=0)
    y_test_combined = np.concatenate((y_test_10, y_test_100), axis=0)

    print("Combined training data shape:", X_train_combined.shape)
    print("Combined testing data shape:", X_test_combined.shape)

    return X_train_combined, y_train_combined, X_test_combined, y_test_combined

def pre_processing(combinedlist):

    X_train_combined = combinedlist[0]
    y_train_combined = combinedlist[1]
    X_test_combined = combinedlist[2]
    y_test_combined = combinedlist[3]

    # Reshape the data
    X_train, X_test = reshape_data(X_train_combined, X_test_combined)

    # Original
    plt.imshow(X_train[1000])
    plt.axis("off")
    plt.title("Original Image")
    plt.show()

    # Grayscale
    img = grayscale(X_train[1000])
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title("Grayscale Image")
    plt.show()

    # Equalize
    img = equalize(img)
    plt.imshow(img)
    plt.show()

    # Normalize
    img = img / 255

    # Gaussian Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
  img = cv2.equalizeHist(img)
  return img

def reshape_data(X_train_combined, X_test_combined):
    X_train = X_train_combined.reshape(X_train_combined.shape[0], 32, 32, 3)
    X_test = X_test_combined.reshape(X_test_combined.shape[0], 32, 32, 3)
    return X_train, X_test

def main():
    combinedlist = load_data()

    img = pre_processing(combinedlist)

main()
