import matplotlib.pyplot as plt
import numpy as np

def plot_augmented_samples(data_generator, n_samples=8):
    
    images, labels = next(iter(data_generator))
    images = images[:n_samples]
    labels = labels[:n_samples]
    
    class_names = {v: k for k, v in data_generator.class_indices.items()}
    
    fig, axes = plt.subplots(2, n_samples // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(n_samples):
        axes[i].imshow(np.squeeze(images[i]), cmap="gray")
        label_index = np.argmax(labels[i])
        axes[i].set_title(class_names[label_index])
        axes[i].axis("off")
    
    plt.suptitle("Augmented Training Samples", fontsize=14)
    plt.tight_layout()
    return fig