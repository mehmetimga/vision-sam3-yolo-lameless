"""
Ensemble methods for combining model predictions
"""
from typing import Dict, List, Any
import numpy as np
from sklearn.linear_model import LogisticRegression


class Ensemble:
    """Ensemble methods for combining predictions"""
    
    @staticmethod
    def voting_ensemble(predictions: Dict[str, float], weights: Dict[str, float] = None) -> float:
        """Weighted voting ensemble"""
        if weights is None:
            weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        weighted_sum = sum(predictions[k] * weights.get(k, 0) for k in predictions.keys())
        total_weight = sum(weights.get(k, 0) for k in predictions.keys())
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    @staticmethod
    def stacking_ensemble(
        base_predictions: np.ndarray,
        meta_learner: Any = None
    ) -> float:
        """Stacking ensemble with meta-learner"""
        if meta_learner is None:
            # Default: simple average
            return float(np.mean(base_predictions))
        
        # Use meta-learner
        predictions_2d = base_predictions.reshape(1, -1)
        return float(meta_learner.predict_proba(predictions_2d)[0, 1])
    
    @staticmethod
    def blending_ensemble(
        predictions: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Blending ensemble with optimized weights"""
        weighted_sum = sum(predictions.get(k, 0.5) * weights.get(k, 0) for k in weights.keys())
        total_weight = sum(weights.values())
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5

