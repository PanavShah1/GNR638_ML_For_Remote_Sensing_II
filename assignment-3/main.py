import os
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tensorflow as tf
import keras
from keras import layers
import tqdm
from time import time
import matplotlib.pyplot as plt
from visualization import visualize
from keras.regularizers import l2

## Config variables
IMAGES_PATH = "../datasets/UCMerced_LandUse/Images"

## Make the required directories
os.makedirs("./cache", exist_ok=True)
os.makedirs("./output", exist_ok=True)


metrics = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "test_accuracy": None}

def create_cnn(
    input_dim=(224,224,3), num_classes = 21
):
    model = keras.Sequential([
        layers.Input(shape=input_dim),

        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
        # layers.Dropout(0.2),

        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
        # layers.Dropout(0.2),

        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
        # layers.Dropout(0.2),

        layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),

        layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)),
        layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),


        # Flatten and Fully Connected Layers
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dense(4096, activation="relu"),
        # layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    # Compile the model
    optimizer = keras.optimizers.Adam(lr=0.005)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


## Read the dataset
if os.path.exists("cache/dataset.pkl"):
    with open("cache/dataset.pkl", "rb") as f:
        data = pickle.load(f)
        categories = data["categories"]
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]
else:
    # Get all categories from the dataset directory
    categories = os.listdir(IMAGES_PATH)

    X = []
    y = []

    # Read images for each category
    for category in tqdm.tqdm(categories, desc="Loading dataset"):
        category_path = os.path.join(IMAGES_PATH, category)
        if os.path.isdir(category_path):
            images = [
                cv2.resize(
                    cv2.cvtColor(
                        cv2.imread(os.path.join(category_path, img_file)),
                        cv2.COLOR_BGR2RGB,
                    ), (224, 224))
                for img_file in os.listdir(category_path)
                if img_file.lower().endswith(".tif")
            ]
            X.extend(images)
            y.extend([category] * len(images))

    # Train => 70%, Validation => 10%, Test => 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, shuffle=True, random_state=42
    )

    # Save the dataset to cache
    with open("cache/dataset.pkl", "wb") as f:
        pickle.dump(
            {
                "categories": categories,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            },
            f,
        )


## Train the classifier
classifier = create_cnn(input_dim=(224,224,3), num_classes=len(categories))
print(classifier.summary())
y_train_ohe = (
    tf.one_hot(np.array([categories.index(y) for y in y_train]), len(categories))
)
start = time()
print("Started training classifier")
X_train = np.array(X_train)
X_val = np.array(X_val)
y_val_ohe = (
    tf.one_hot(np.array([categories.index(y) for y in y_val]), len(categories))
)
history = classifier.fit(
    X_train,
    y_train_ohe,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val_ohe),
)
print(f"Classifier trained in {time() - start} seconds")

metrics["train_loss"] = history.history["loss"]
metrics["train_accuracy"] = history.history["accuracy"]
metrics["val_loss"] = history.history["val_loss"]
metrics["val_accuracy"] = history.history["val_accuracy"]

# Test
X_test = np.array(X_test)
y_test_ohe = (
    tf.one_hot(np.array([categories.index(y) for y in y_test]), len(categories))
)
test_preds = classifier.predict(X_test)
test_preds = np.argmax(test_preds, axis=1)
test_accuracy = np.mean(test_preds == np.array([categories.index(y) for y in y_test]))

metrics["test_accuracy"] = test_accuracy

# print(f"Validation accuracy: {val_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Saving output
with open(f"output/output_cnn.txt", "w") as f:
    # f.write(f"Validation accuracy: {val_accuracy}\n")
    f.write(str(metrics))
