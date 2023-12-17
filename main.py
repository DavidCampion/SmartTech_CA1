import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from keras.src.layers import BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_data():
    # Load CIFAR-10 data
    (X_train_10, y_train_10), (X_test_10, y_test_10) = tf.keras.datasets.cifar10.load_data()
    y_train_10 = y_train_10.flatten()
    y_test_10 = y_test_10.flatten()

    # Load CIFAR-100 data
    (X_train_100, y_train_100), (X_test_100, y_test_100) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    y_train_100 = y_train_100.flatten()
    y_test_100 = y_test_100.flatten()

    # Filter classes
    cifar10_classes = [1, 2, 3, 4, 5, 7, 9]
    cifar100_classes = [2, 8, 11, 13, 17, 19, 34, 35, 41, 46, 48, 58, 65, 80, 89, 90, 98]

    # ref: https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html#cifar10-load
    mask_10 = np.isin(y_train_10, cifar10_classes)
    X_train_10 = X_train_10[mask_10]
    y_train_10 = y_train_10[mask_10]

    mask_10_test = np.isin(y_test_10, cifar10_classes)
    X_test_10 = X_test_10[mask_10_test]
    y_test_10 = y_test_10[mask_10_test]

    mask_100 = np.isin(y_train_100, cifar100_classes)
    X_train_100 = X_train_100[mask_100]
    y_train_100 = y_train_100[mask_100]

    mask_100_test = np.isin(y_test_100, cifar100_classes)
    X_test_100 = X_test_100[mask_100_test]
    y_test_100 = y_test_100[mask_100_test]

    # Combine and return the datasets
    X_train_combined = np.concatenate((X_train_10, X_train_100), axis=0)
    y_train_combined = np.concatenate((y_train_10, y_train_100), axis=0)
    X_test_combined = np.concatenate((X_test_10, X_test_100), axis=0)
    y_test_combined = np.concatenate((y_test_10, y_test_100), axis=0)

    print("Total number of training images:", X_train_combined.shape[0])
    print("Total number of testing images:", X_test_combined.shape[0])

    return X_train_combined, y_train_combined, X_test_combined, y_test_combined

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

def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img.reshape(32, 32, 1)  # Reshape for single channel

def pre_processing(combinedlist):
    X_train_combined = combinedlist[0]

    # Original
    plt.imshow(X_train_combined[1000])
    plt.axis("off")
    plt.title("Original Image")
    plt.show()

    # Grayscale
    img = grayscale(X_train_combined[1000])
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

def data_exploration(combinedlist):
    X_train_combined = combinedlist[0]
    y_train_combined = combinedlist[1]

    classes = [1, 2, 3, 4, 5, 7, 9, 8, 11, 13, 17, 19, 34, 35, 41, 46, 48, 58, 65, 80, 89, 90, 98]

    filtered_classes = np.intersect1d(classes, np.unique(y_train_combined))

    num_classes = len(filtered_classes)
    cols = 5
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
    fig.tight_layout()

    num_of_samples = []

    for i in range(cols):
        for idx, j in enumerate(filtered_classes):
            x_selected = X_train_combined[y_train_combined == j]
            if len(x_selected) > 0:
                axs[idx][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap('gray'))
            axs[idx][i].axis("off")
            if i == 2:
                num_of_samples.append(len(x_selected))
                axs[idx][i].set_title(str(j))

    print("Number of images per class:", num_of_samples)

    plt.show()

    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training set")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

    # Printing num_of_samples
    print(num_of_samples)

# https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/
# https://keras.io/api/layers/normalization_layers/batch_normalization/
def leNet_model(num_classes):
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(125, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Block
    model.add(Conv2D(250, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train_preprocessed, y_train_combined, X_test_preprocessed, Y_test_combined):
    history = model.fit(X_train_preprocessed, y_train_combined, epochs=15, validation_data=(X_test_preprocessed, Y_test_combined), batch_size=400, verbose=1, shuffle=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(["Training", "Validation"])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.show()

def main():
    X_train_combined, y_train_combined, X_test_combined, Y_test_combined = load_data()
    data_exploration([X_train_combined, y_train_combined])

    # Apply preprocessing
    X_train_preprocessed = np.array([preprocess(img) for img in X_train_combined])
    X_test_preprocessed = np.array([preprocess(img) for img in X_test_combined])

    # Demo Of PreProcessing
    pre_processing([X_train_combined, y_train_combined])

    # Define the model
    num_classes = np.max(y_train_combined) + 1
    model = leNet_model(num_classes)

    # Print the model summary
    print(model.summary())

    # image data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(X_train_preprocessed)
    # Train the model
    train_model(model, X_train_preprocessed, y_train_combined, X_test_preprocessed, Y_test_combined)

main()
