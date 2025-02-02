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

## Config variables
IMAGES_PATH = "../datasets/UCMerced_LandUse/Images"

## Make the required directories
os.makedirs("./cache", exist_ok=True)
os.makedirs("./output", exist_ok=True)


def create_bow_classifier(
    input_dim=72*72, hidden_dim1=1024, hidden_dim2=128, num_classes=21
):
    model = keras.Sequential(
        [
            layers.Input((input_dim,)),
            # First layer
            layers.Dense(hidden_dim1, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # Second layer
            layers.Dense(hidden_dim2, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            # Output layer
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
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
                        cv2.COLOR_BGR2GRAY,
                    ), (72, 72))
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
classifier = create_bow_classifier(input_dim=72*72, num_classes=len(categories))
y_train_ohe = (
    tf.one_hot(np.array([categories.index(y) for y in y_train]), len(categories)),
)
start = time()
print("Started training classifier")
X_train = np.array(X_train).reshape(len(X_train), -1)  # Flatten each image
X_val = np.array(X_val).reshape(len(X_val), -1)
y_val_ohe = (
    tf.one_hot(np.array([categories.index(y) for y in y_val]), len(categories)),
)
classifier.fit(
    X_train,
    y_train_ohe,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val_ohe),
)
print(f"Classifier trained in {time() - start} seconds")

# Test
X_test = np.array(X_test).reshape(len(X_test), -1)
y_test_ohe = (
    tf.one_hot(np.array([categories.index(y) for y in y_test]), len(categories)),
)
test_preds = classifier.predict(X_test)
test_preds = np.argmax(test_preds, axis=1)
test_accuracy = np.mean(test_preds == np.array([categories.index(y) for y in y_test]))

# print(f"Validation accuracy: {val_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Saving output
with open(f"output/output_mlp2.txt", "w") as f:
    # f.write(f"Validation accuracy: {val_accuracy}\n")
    f.write(f"Test accuracy: {test_accuracy}\n")
