import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold
import logging

logger = logging.getLogger(__name__)

def evaluate_knn(data: pd.DataFrame,
                 k_values: List[int] = list(range(1, 16)),
                 distance_metric: str = 'euclidean',
                 n_splits: int = 10,
                 random_state: int = 2) -> Dict[int, List[Dict[str, np.ndarray]]]:
    """Evaluates KNN classifier performance for different values of k using cross-validation."""
    # Prepare data
    X = np.stack(data["embedding"].values)
    y = data["syndrome_id"].values
    groups = data["subject_id"].values

    # Initialize cross-validator
    cv = StratifiedGroupKFold(n_splits=n_splits, 
                              shuffle=True, 
                              random_state=random_state)
    
    results = {}
    for k in k_values:
        logger.info(f"Evaluating KNN with k={k} and metric={distance_metric}")
        model = KNeighborsClassifier(n_neighbors=k, 
                                     metric=distance_metric)
        fold_results = []
        
        for train_idx, test_idx in cv.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)

            y_predict = model.predict(X_test)
            y_probabilities = model.predict_proba(X_test)

            fold_results.append({
                "y_true": y_test,
                "y_predict": y_predict,
                "y_probabilities": y_probabilities,
                'labels': model.classes_
            })
        
        results[k] = fold_results
    
    return results

if __name__ == "__main__":
    from processing import flatten_data
    logging.basicConfig(level=logging.INFO)

    # Example usage
    df = flatten_data('data/mini_gm_public_v0.1.p')
    knn_results = evaluate_knn(df, distance_metric='cosine')
    print("Done.")