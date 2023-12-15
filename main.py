import pickle

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random
from keras.datasets import cifar10, cifar100


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_data():
    file_10 = 'C:\\SmartTech_CA1_2\\data\\cifar-10-batches-py\\data_batch_1'
    file_10_test = 'C:\\SmartTech_CA1_2\\data\\cifar-10-batches-py\\test_batch'
    file_100 = 'C:\\SmartTech_CA1_2\\data\\cifar-100-python\\train'
    file_100_test = 'C:\\SmartTech_CA1_2\\data\\cifar-100-python\\test'

    data_batch_10 = unpickle(file_10)
    data_batch_100 = unpickle(file_100)
    data_batch_10_test = unpickle(file_10_test)
    data_batch_100_test = unpickle(file_100_test)
    print(data_batch_10.keys())
    print(data_batch_100.keys())


    cifar10_classes = [1, 2, 3, 4, 5, 7, 9]
    cifar100_classes = [2, 8, 11, 13, 17, 19, 34, 35, 41, 46, 48, 58, 65, 80, 89, 90, 98]

    # ref: https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html#cifar10-load
    mask_10 = np.isin(data_batch_10['labels'], cifar10_classes)
    X_train_10 = data_batch_10['data'][mask_10].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train_10 = np.array(data_batch_10['labels'])[mask_10]

    mask_10_test = np.isin(data_batch_10_test['labels'], cifar10_classes)
    X_test_10 = data_batch_10_test['data'][mask_10_test].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test_10 = np.array(data_batch_10_test['labels'])[mask_10_test]

    mask_100 = np.isin(data_batch_100['fine_labels'], cifar100_classes)
    X_train_100 = data_batch_100['data'][mask_100].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train_100 = np.array(data_batch_100['fine_labels'])[mask_100]

    mask_100_test = np.isin(data_batch_100_test['fine_labels'], cifar100_classes)
    X_test_100 = data_batch_100_test['data'][mask_100_test].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test_100 = np.array(data_batch_100_test['fine_labels'])[mask_100_test]

    X_train_combined = np.concatenate((X_train_10, X_train_100), axis=0)
    y_train_combined = np.concatenate((y_train_10, y_train_100), axis=0)
    X_test_combined = np.concatenate((X_test_10, X_test_100), axis=0)
    y_test_combined = np.concatenate((y_test_10, y_test_100), axis=0)

    print("Filtered CIFAR-10 Data Shape(x):", X_train_10.shape)
    print("Filtered CIFAR-100 Data Shape(x):", X_train_100.shape)
    print("Filtered CIFAR-10 Labels Shape(y):", y_train_10.shape)
    print("Filtered CIFAR-100 Labels Shape(y):", y_train_100.shape)
    print("Combined Training Data Shape(x):", X_train_combined.shape)
    print("Combined Training Labels Shape(y):", y_train_combined.shape)
    print("Combined Testing Data Shape(x):", X_test_combined.shape)
    print("Combined Testing Labels Shape(y):", y_test_combined.shape)

    random_index = random.randint(0, len(X_train_100) - 1)
    image = X_train_100[random_index]
    label = y_train_100[random_index]

    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.show()
    return X_train_combined,y_train_combined
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
    load_data()
    # combinedlist = load_data()

#   img = pre_processing(combinedlist)

main()
