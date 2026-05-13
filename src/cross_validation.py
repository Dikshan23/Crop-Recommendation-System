"""
src/cross_validation.py
=======================
From-scratch stratified k-fold cross-validation utilities.
"""
import numpy as np
from collections import defaultdict


def stratified_kfold(X, y, n_splits=5, random_state=42):
    """
    Generate stratified k-fold indices.
    
    Args:
        X: Feature array
        y: Label array
        n_splits: Number of folds
        random_state: Random seed for reproducibility
    
    Yields:
        train_idx, val_idx: Indices for each fold
    """
    rng = np.random.RandomState(random_state)
    
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_to_indices[label].append(idx)
    
    # Shuffle indices within each class
    for label in class_to_indices:
        indices = np.array(class_to_indices[label])
        rng.shuffle(indices)
        class_to_indices[label] = indices
    
    # Create folds for each class
    folds = [[] for _ in range(n_splits)]
    for label, indices in class_to_indices.items():
        # Split class indices into n_splits roughly equal parts
        fold_sizes = np.full(n_splits, len(indices) // n_splits, dtype=int)
        fold_sizes[:len(indices) % n_splits] += 1
        
        current = 0
        for fold_idx, size in enumerate(fold_sizes):
            folds[fold_idx].extend(indices[current:current + size])
            current += size
    
    # Yield train/val splits
    for i in range(n_splits):
        val_idx = np.array(folds[i])
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, val_idx


def cross_validate(model_class, X, y, param_grid, n_splits=5, random_state=42):
    """
    Perform cross-validation grid search.
    
    Args:
        model_class: Model class (not instantiated)
        X, y: Training data
        param_grid: Dict of parameter lists to search
        n_splits: Number of CV folds
        random_state: Random seed
    
    Returns:
        best_params: Dict of best parameters
        best_score: Mean CV accuracy
        cv_results: List of (params, fold_scores) for all combinations
    """
    cv_results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    from itertools import product
    param_combinations = [dict(zip(param_names, combo)) 
                         for combo in product(*param_values)]
    
    for params in param_combinations:
        fold_scores = []
        
        for train_idx, val_idx in stratified_kfold(X, y, n_splits, random_state):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model with these parameters
            model = model_class(**params)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            y_pred = model.predict(X_val_fold)
            accuracy = np.mean(y_pred == y_val_fold)
            fold_scores.append(accuracy)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        cv_results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores
        })
        
        print(f"Params: {params} | Mean CV: {mean_score:.4f} (±{std_score:.4f})")
    
    # Find best parameters
    best_result = max(cv_results, key=lambda x: x['mean_score'])
    
    return best_result['params'], best_result['mean_score'], cv_results