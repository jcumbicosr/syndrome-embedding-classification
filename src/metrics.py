import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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

def f1_score(y_true: np.ndarray, 
             y_predict: np.ndarray) -> float:
    """Calculates the F1 score."""
    labels = np.unique(y_true)
    f1_scores = []
    for c in labels:
        # Calculate TP, FP, FN for the current class 'c'
        tp = np.sum((y_true == c) & (y_predict == c))
        fp = np.sum((y_true != c) & (y_predict == c))
        fn = np.sum((y_true == c) & (y_predict != c))
        
        # Calculate Precision and Recall with safe division (avoiding division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Calculate F1-Score for class 'c'
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        f1_scores.append(f1)
        
    return float(np.mean(f1_scores))

def multiclass_auc(y_true: np.ndarray, 
                   y_probabilities: np.ndarray,
                   classes: np.ndarray) -> float:
    """Calculates the multiclass AUC using the One-vs-Rest approach."""
    y_true_binarized = label_binarize(y_true, classes=classes)
    
    # Calculate AUC for each class
    auc_scores = []
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_probabilities[:, i])
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)

    return float(np.mean(auc_scores))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    y_true = np.array([0, 1, 2])
    y_predict = np.array([1, 1, 2])
    y_probabilities = np.array([[0.1, 0.7, 0.2],
                                [0.3, 0.4, 0.3],
                                [0.2, 0.2, 0.6]])
    classes = np.array([0, 1, 2])
    k = 2

    # accuracy = top_k_accuracy(y_true, y_probabilities, classes, k)
    # print(f"Top-{k} Accuracy: {accuracy:.4f}")

    # f1_score_value = f1_score(y_true, y_predict)
    # print(f"F1 Score: {f1_score_value:.4f}")

    auc_value = multiclass_auc(y_true, y_probabilities, classes)
    print(f"Multiclass AUC: {auc_value:.4f}")

