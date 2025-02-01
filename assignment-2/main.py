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
VOCAB_SIZE = 250
N_NEIGHBORS = 5

## Make the required directories
os.makedirs("./cache", exist_ok=True)
os.makedirs("./output", exist_ok=True)

## Initialize SIFT
SIFT = cv2.SIFT_create()  # type: ignore


## Helper functions
def get_img_keypoints(im: np.ndarray) -> np.ndarray:
    _, des = SIFT.detectAndCompute(im, None)
    return des if des is not None else np.array([])


def get_bow_representation(im: np.ndarray) -> np.ndarray:
    _, des = SIFT.detectAndCompute(im, None)
    if des is None:
        return np.zeros(VOCAB_SIZE, dtype=np.float32)

    bow = np.histogram(
        vocab_model.predict(des),
        bins=range(VOCAB_SIZE + 1),
    )[0]
    bow = bow.astype(np.float32)
    bow = bow / np.sum(bow)
    return bow


def create_bow_classifier(
    input_dim=250, hidden_dim1=128, hidden_dim2=64, num_classes=21
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
if os.path.exists("../assignment-1/cache/dataset.pkl"):
    with open("../assignment-1/cache/dataset.pkl", "rb") as f:
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
                cv2.cvtColor(
                    cv2.imread(os.path.join(category_path, img_file)),
                    cv2.COLOR_BGR2GRAY,
                )
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
    with open("../assignment-1/cache/dataset.pkl", "wb") as f:
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


## Find all the keypoints
if os.path.exists("../assignment-1/cache/all_keypoints.pkl"):
    with open("../assignment-1/cache/all_keypoints.pkl", "rb") as f:
        all_keypoints = pickle.load(f)
else:
    all_keypoints = []
    for im in tqdm.tqdm(X_train, desc="Finding keypoints in training set"):
        all_keypoints.extend(get_img_keypoints(im))
    all_keypoints = np.array(all_keypoints)
    with open("../assignment-1/cache/all_keypoints.pkl", "wb") as f:
        pickle.dump(all_keypoints, f)

## Generate a vocabulary
if os.path.exists(f"../assignment-1/cache/vocab_model_{VOCAB_SIZE}.pkl"):
    with open(f"../assignment-1/cache/vocab_model_{VOCAB_SIZE}.pkl", "rb") as f:
        vocab_model = pickle.load(f)
else:
    vocab_model = KMeans(n_clusters=VOCAB_SIZE + 1, random_state=42)
    print("Started training KMeans model")
    start = time()
    vocab_model.fit(all_keypoints)
    print(f"KMeans model trained in {time() - start} seconds")
    with open(f"cache/vocab_model_{VOCAB_SIZE}.pkl", "wb") as f:
        pickle.dump(vocab_model, f)

bows, val_bows, test_bows = None, None, None
## Generate the Bag of Words representation of the images
if os.path.exists(f"../assignment-1/cache/train_bows_{VOCAB_SIZE}.pkl"):
    with open(f"../assignment-1/cache/train_bows_{VOCAB_SIZE}.pkl", "rb") as f:
        bows = pickle.load(f)
else:
    bows = np.array(
        [
            get_bow_representation(im)
            for im in tqdm.tqdm(X_train, desc="Generating BoWs for training set")
        ]
    )
    with open(f"../assignment-1/cache/train_bows_{VOCAB_SIZE}.pkl", "wb") as f:
        pickle.dump(bows, f)

## Train the classifier
classifier = create_bow_classifier(input_dim=VOCAB_SIZE, num_classes=len(categories))
bows = tf.convert_to_tensor(bows, dtype=tf.float32)
y_train_ohe = (
    tf.one_hot(np.array([categories.index(y) for y in y_train]), len(categories)),
)
start = time()
print("Started training classifier")
classifier.fit(
    bows,
    y_train_ohe,
    epochs=10,
    batch_size=32,
)
print(f"Classifier trained in {time() - start} seconds")

## Evaluate the classifier
# Validation
if os.path.exists(f"../assignment-1/cache/val_bows_{VOCAB_SIZE}.pkl"):
    with open(f"../assignment-1/cache/val_bows_{VOCAB_SIZE}.pkl", "rb") as f:
        val_bows = pickle.load(f)
else:
    val_bows = np.array(
        [
            get_bow_representation(im)
            for im in tqdm.tqdm(X_val, desc="Generating BoWs for validation set")
        ]
    )
    with open(f"../assignment-1/cache/val_bows_{VOCAB_SIZE}.pkl", "wb") as f:
        pickle.dump(val_bows, f)

val_preds = classifier.predict(val_bows)
val_preds = np.argmax(val_preds, axis=1)
val_accuracy = np.mean(val_preds == np.array([categories.index(y) for y in y_val]))

# Test
if os.path.exists(f"../assignment-1/cache/test_bows_{VOCAB_SIZE}.pkl"):
    with open(f"../assignment-1/cache/test_bows_{VOCAB_SIZE}.pkl", "rb") as f:
        test_bows = pickle.load(f)
else:
    test_bows = np.array(
        [
            get_bow_representation(im)
            for im in tqdm.tqdm(X_test, desc="Generating BoWs for test set")
        ]
    )
    with open(f"../assignment-1/cache/test_bows_{VOCAB_SIZE}.pkl", "wb") as f:
        pickle.dump(test_bows, f)

test_preds = classifier.predict(test_bows)
test_preds = np.argmax(test_preds, axis=1)
test_accuracy = np.mean(test_preds == np.array([categories.index(y) for y in y_test]))

print(f"Validation accuracy: {val_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Saving output
with open(f"output/output_mlp1_{VOCAB_SIZE}.txt", "w") as f:
    f.write(f"Validation accuracy: {val_accuracy}\n")
    f.write(f"Test accuracy: {test_accuracy}\n")

fig = visualize(bows, y_train)
fig.savefig(f"output/visualization_mlp1_{VOCAB_SIZE}.png")
# plt.show()
