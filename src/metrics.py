import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from typing import Dict, List
from pathlib import Path

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
                     topk_values: List[int],
                     tag: str,
                     output_dir: str = "reports/tables/"
) -> pd.DataFrame:
    """Evaluates metrics for each value of k."""
    summary = {}
    for k, fold_results in results.items():
        f1_scores = []
        auc_scores = []
        topk_scores = {tk: [] for tk in topk_values}
        
        for fold in fold_results:
            y_true = fold["y_true"]
            y_predict = fold["y_predict"]
            y_probabilities = fold["y_probabilities"]
            classes = fold['labels']

            f1_scores.append(f1_score(y_true, y_predict))
            auc_scores.append(multiclass_auc(y_true, y_probabilities, classes))
            for tk in topk_values:
                topk_scores[tk].append(top_k_accuracy(y_true, y_probabilities, classes, tk))

        summary[k] = {
            "f1_score": float(np.mean(f1_scores)),
            "auc_score": float(np.mean(auc_scores)),
            **{f"topk_{tk}": float(np.mean(topk_scores[tk])) for tk in topk_values}
        }

    # Convert to DataFrame for better visualization
    summary_df = pd.DataFrame.from_dict(summary, orient='index')  
    summary_df.index.name = 'k'

    # Save summary table
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path = output_path / f"metrics_summary_{tag}.csv"
    summary_df.to_csv(table_path)
    logger.info(f"Metrics summary saved to: {table_path}")

    return summary_df

def plot_best_k(
    results: pd.DataFrame,
    tag: str,
    metrics: List[str] = None,
    output_dir: str = "reports/figures/",
    show: bool = False
) -> Dict[str, int]:
    """
    Plots performance vs k and returns the best k value.
    """
    # Select the metrics to plot
    plot_data = results
    if metrics is not None:
        plot_data = plot_data[metrics]

    # Get the best k value
    df_best = pd.DataFrame({
        "best_value": plot_data.max(),
        "best_k": plot_data.idxmax(),
    })
    for row in df_best.itertuples():
        logger.info(f"Best k for {row.Index}: {row.best_k} with value {row.best_value:.4f}")


    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    results.plot(ax=ax, marker='o', linewidth=2)

    # Highlight best k for each metric
    colors = [line.get_color() for line in ax.get_lines()]
    for (metric, row), color in zip(df_best.iterrows(), colors):
        ax.scatter(
            row["best_k"], row["best_value"],
            color=color, s=100, zorder=5, edgecolors="black", linewidths=1.5
        )

    # Update legend with best k info
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        f"{label} (k={int(df_best.loc[label, 'best_k'])}, {df_best.loc[label, 'best_value']:.2f})"
        for label in labels
    ]
    ax.legend(handles, new_labels, fontsize=10)
    
    # Format axes
    ax.set_xlabel("Number of Neighbors (k)", fontsize=14)
    ax.set_title("KNN Performance vs k", fontsize=16)
    ax.grid(alpha=0.3)

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"metrics_vs_k_{tag}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    logger.info(f"Best k plot saved to: {fig_path}")

    if show:
        plt.show()
    
    return df_best["best_k"].to_dict()

def average_roc_curve(fold_results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Calculates the average ROC curve for multiclass classification."""
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for fold in fold_results:
        y_true = fold["y_true"]
        y_proba = fold["y_probabilities"]
        classes = fold["labels"]

        # Binarize labels for multi-class ROC calculation
        y_true_bin = label_binarize(y_true, classes=classes)

        # Compute micro-average ROC curve for this specific fold
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())

        # Interpolate the True Positive Rate
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    
    # Average the True Positive Rates and AUCs across all folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    return mean_fpr, mean_tpr, mean_auc

def plot_comparative_roc(results_euclidean: Dict[int, List[Dict[str, np.ndarray]]], 
                         results_cosine: Dict[int, List[Dict[str, np.ndarray]]],
                         best_k_euclidean: int,
                         best_k_cosine: int,
                         output_dir: str = "reports/figures/",
                         show: bool = False
) -> None:
    """Plots comparative ROC curves for Euclidean and Cosine distance metrics."""
    fpr_cos, tpr_cos, auc_cos = average_roc_curve(results_cosine[best_k_cosine])
    fpr_euc, tpr_euc, auc_euc = average_roc_curve(results_euclidean[best_k_euclidean])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr_cos, tpr_cos, color='darkorange', lw=2,
            label=f'Cosine (k={best_k_cosine}, Averaged AUC = {auc_cos:.3f})')
    ax.plot(fpr_euc, tpr_euc, color='navy', lw=2,
            label=f'Euclidean (k={best_k_euclidean}, Averaged AUC = {auc_euc:.3f})')
    
    # Add the random chance diagonal line
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Chance')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Averaged Cross-Validation ROC Curves: Cosine vs Euclidean', fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "comparative_roc.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Comparative ROC plot saved to: {fig_path}")

    if show:
        plt.show()
    

if __name__ == "__main__":
    from processing import flatten_data
    from classifier import evaluate_knn
    logging.basicConfig(level=logging.INFO)

    df = flatten_data('data/mini_gm_public_v0.1.p')
    results_cosine = evaluate_knn(df, distance_metric='cosine')
    cosine_summary = evaluate_metrics(results_cosine, topk_values=[1, 3, 5], tag="cosine")
    best_k_cosine = plot_best_k(cosine_summary, tag="cosine")

    results_euclidean = evaluate_knn(df, distance_metric='euclidean')
    euclidean_summary = evaluate_metrics(results_euclidean, topk_values=[1, 3, 5], tag="euclidean")
    best_k_euclidean = plot_best_k(euclidean_summary, tag="euclidean")

    plot_comparative_roc(results_euclidean=results_euclidean, 
                         results_cosine=results_cosine, 
                         best_k_euclidean=best_k_euclidean['auc_score'], 
                         best_k_cosine=best_k_cosine['auc_score'])


    print("Done.")

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

