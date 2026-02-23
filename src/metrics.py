import numpy as np

import logging


logger = logging.getLogger(__name__)

def top_k_accuracy(y_true: np.ndarray, 
                   y_probabilities: np.ndarray,
                   classes: np.ndarray, 
                   k: int) -> float:
    """Calculates the top-k accuracy."""
    top_k_indices = np.argsort(y_probabilities, axis=1)[:, -k:]
    top_k_classes = classes[top_k_indices]

    # Check if the true label exists within the top k predicted classes
    hits = np.any(top_k_classes == y_true[:, np.newaxis], axis=1)
    return float(np.mean(hits))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    y_true = np.array([0, 1, 2])
    y_probabilities = np.array([[0.1, 0.7, 0.2],
                                [0.3, 0.4, 0.3],
                                [0.2, 0.2, 0.6]])
    classes = np.array([0, 1, 2])
    k = 2

    accuracy = top_k_accuracy(y_true, y_probabilities, classes, k)
    print(f"Top-{k} Accuracy: {accuracy:.4f}")
