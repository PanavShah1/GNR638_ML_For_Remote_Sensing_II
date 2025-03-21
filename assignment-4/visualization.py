from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def convert_labels_to_numbers(labels):
    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    number_map = {i: label for i, label in enumerate(unique_labels)}
    return {"values": np.array([label_map[label] for label in labels]), "label_map": label_map, "number_map": number_map}

def visualize(bows, y_train):
    fig = plt.figure(figsize=(15, 10))

    # Convert labels to numeric values and mappings
    y_train_converted = convert_labels_to_numbers(y_train)
    numeric_labels = y_train_converted["values"]
    number_map = y_train_converted["number_map"]
    unique_labels = np.unique(numeric_labels)
    legend_labels = [number_map[label] for label in unique_labels]

    # --- TSNE 2D Visualization ---
    tsne_2d = TSNE(n_components=2, random_state=42)
    tsne_bows_2d = tsne_2d.fit_transform(bows)
    ax1 = plt.subplot(2, 2, 1)
    scatter_2d_tsne = ax1.scatter(tsne_bows_2d[:, 0], tsne_bows_2d[:, 1], c=numeric_labels, cmap='viridis')
    ax1.set_title("2D TSNE Visualization")

    # --- TSNE 3D Visualization ---
    tsne_3d = TSNE(n_components=3, random_state=42)
    tsne_bows_3d = tsne_3d.fit_transform(bows)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    scatter_3d_tsne = ax2.scatter(
        tsne_bows_3d[:, 0], tsne_bows_3d[:, 1], tsne_bows_3d[:, 2], c=numeric_labels, cmap='viridis'
    )
    ax2.set_title("3D TSNE Visualization")
    ax2.set_xlabel("TSNE Component 1")
    ax2.set_ylabel("TSNE Component 2")
    ax2.set_zlabel("TSNE Component 3")

    # --- PCA 2D Visualization ---
    pca_2d = PCA(n_components=2)
    pca_bows_2d = pca_2d.fit_transform(bows)
    ax3 = plt.subplot(2, 2, 3)
    scatter_2d_pca = ax3.scatter(pca_bows_2d[:, 0], pca_bows_2d[:, 1], c=numeric_labels, cmap='viridis')
    ax3.set_xlabel("PCA Component 1")
    ax3.set_ylabel("PCA Component 2")
    ax3.set_title("2D PCA Visualization")

    # --- PCA 3D Visualization ---
    pca_3d = PCA(n_components=3)
    pca_bows_3d = pca_3d.fit_transform(bows)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    scatter_3d_pca = ax4.scatter(
        pca_bows_3d[:, 0], pca_bows_3d[:, 1], pca_bows_3d[:, 2], c=numeric_labels, cmap='viridis'
    )
    ax4.set_xlabel("PCA Component 1")
    ax4.set_ylabel("PCA Component 2")
    ax4.set_zlabel("PCA Component 3")
    ax4.set_title("3D PCA Visualization")

    # Create a single legend on the left
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=scatter_2d_tsne.cmap(scatter_2d_tsne.norm(label)))
        for label in unique_labels
    ]
    fig.subplots_adjust(left=0.3)  
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(0.05, 0.5), title="Classes", fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0.2, 0, 1, 1])  
    # plt.show()

    return fig
