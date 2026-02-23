from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from typing import Dict, List

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

def evaluate_metrics(results: Dict[int, List[Dict[str, np.ndarray]]], 
                     topk_value: int,) -> Dict[int, Dict[str, float]]:
    """Evaluates metrics for each value of k."""
    summary = {}
    for k, fold_results in results.items():
        topk_score = []
        f1_scores = []
        auc_scores = []
        
        for fold in fold_results:
            y_true = fold["y_true"]
            y_predict = fold["y_predict"]
            y_probabilities = fold["y_probabilities"]
            classes = fold['labels']

            topk_score.append(top_k_accuracy(y_true, y_probabilities, classes, topk_value))
            f1_scores.append(f1_score(y_true, y_predict))
            auc_scores.append(multiclass_auc(y_true, y_probabilities, classes))

        summary[k] = {
            "topk_score": float(np.mean(topk_score)),
            "f1_score": float(np.mean(f1_scores)),
            "auc_score": float(np.mean(auc_scores))}
    
    return summary

def plot_best_k(
    results: Dict[int, Dict[str, float]],
    metric: str = "f1_score",
    output_dir: str = "reports/figures/",
    show: bool = False
) -> int:
    """
    Plots performance vs k and returns the best k value.
    """
    x = sorted(results.keys())
    y = [results[k][metric] for k in x]

    # Get the best k value
    best_k = x[np.argmax(y)]
    best_metric = max(y)
    logger.info(f"Best k: {best_k} with {metric}: {best_metric}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker='o', linewidth=2)
    ax.scatter(best_k, best_metric, s=100)

    ax.set_xlabel("Number of Neighbors (k)", fontsize=14)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=14)
    ax.set_title("KNN Performance vs k", fontsize=16)
    ax.grid(alpha=0.3)

    # Highlight best k
    ax.annotate(
        f"Best k = {best_k}\n{metric.replace('_', ' ').title()} = {best_metric:.2f}",
        xy=(best_k, best_metric),
        xytext=(best_k * 0.98, best_metric * 0.98),
    )

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / f"knn_{metric}_vs_k.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    logger.info(f"Best k plot saved to: {fig_path}")

    if show:
        plt.show()
    
    return best_k


if __name__ == "__main__":
    from processing import flatten_data
    from classifier import evaluate_knn
    logging.basicConfig(level=logging.INFO)

    df = flatten_data('data/mini_gm_public_v0.1.p')
    knn_results = evaluate_knn(df, distance_metric='cosine')
    metrics_summary = evaluate_metrics(knn_results, topk_value=3)
    best_k = plot_best_k(metrics_summary, metric="f1_score")

    # Example usage
    # y_true = np.array([0, 1, 2])
    # y_predict = np.array([1, 1, 2])
    # y_probabilities = np.array([[0.1, 0.7, 0.2],
    #                             [0.3, 0.4, 0.3],
    #                             [0.2, 0.2, 0.6]])
    # classes = np.array([0, 1, 2])
    # k = 2

    # accuracy = top_k_accuracy(y_true, y_probabilities, classes, k)
    # print(f"Top-{k} Accuracy: {accuracy:.4f}")

    # f1_score_value = f1_score(y_true, y_predict)
    # print(f"F1 Score: {f1_score_value:.4f}")

    # auc_value = multiclass_auc(y_true, y_probabilities, classes)
    # print(f"Multiclass AUC: {auc_value:.4f}")

