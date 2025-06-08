"""
Uncertainty Estimation for Physics-Based Features
Provides uncertainty quantification and confidence calibration for Bayesian analysis
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class UncertaintyType(Enum):
    """Types of uncertainty"""
    EPISTEMIC = "epistemic"  # Model uncertainty
    ALEATORIC = "aleatoric"  # Data uncertainty
    TOTAL = "total"          # Combined uncertainty

@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates"""
    mean: float
    variance: float
    std: float
    confidence_interval: Tuple[float, float]
    uncertainty_type: UncertaintyType
    confidence_level: float = 0.95
    
    @property
    def lower_bound(self) -> float:
        return self.confidence_interval[0]
    
    @property
    def upper_bound(self) -> float:
        return self.confidence_interval[1]
    
    @property
    def interval_width(self) -> float:
        return self.upper_bound - self.lower_bound

class PhysicsUncertaintyEstimator:
    """
    Estimates uncertainty in physics-based features for deepfake detection
    
    Provides functionality for:
    - Epistemic uncertainty estimation (model uncertainty)
    - Aleatoric uncertainty estimation (data uncertainty)
    - Confidence calibration
    - Uncertainty propagation
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 bootstrap_samples: int = 1000,
                 enable_calibration: bool = True):
        """
        Initialize uncertainty estimator
        
        Args:
            confidence_level: Default confidence level for intervals
            bootstrap_samples: Number of bootstrap samples for uncertainty estimation
            enable_calibration: Whether to enable confidence calibration
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.enable_calibration = enable_calibration
        
        # Calibration parameters (learned from data)
        self.calibration_params = {
            'delta_fr': {'scale': 1.0, 'shift': 0.0},
            'delta_ft': {'scale': 1.0, 'shift': 0.0},
            'delta_fv': {'scale': 1.0, 'shift': 0.0}
        }
        
        # Historical statistics for uncertainty estimation
        self.feature_statistics = {
            'delta_fr': {'mean': 7.0, 'std': 1.5, 'samples': []},
            'delta_ft': {'mean': 0.08, 'std': 0.03, 'samples': []},
            'delta_fv': {'mean': 1.5, 'std': 0.8, 'samples': []}
        }
        
        logger.info(f"PhysicsUncertaintyEstimator initialized: confidence={confidence_level}, bootstrap={bootstrap_samples}")
    
    def estimate_epistemic_uncertainty(self, 
                                     feature_values: Dict[str, float],
                                     model_ensemble: Optional[List] = None) -> Dict[str, UncertaintyEstimate]:
        """
        Estimate epistemic uncertainty (model uncertainty)
        
        Args:
            feature_values: Dictionary of physics feature values
            model_ensemble: Optional ensemble of models for uncertainty estimation
            
        Returns:
            Dictionary of uncertainty estimates for each feature
        """
        uncertainties = {}
        
        for feature_name, value in feature_values.items():
            if feature_name not in self.feature_statistics:
                # Use default uncertainty for unknown features
                uncertainty = self._default_epistemic_uncertainty(value)
            else:
                # Estimate based on historical statistics
                stats = self.feature_statistics[feature_name]
                
                # Model uncertainty based on distance from typical values
                distance_from_mean = abs(value - stats['mean'])
                normalized_distance = distance_from_mean / max(stats['std'], 1e-6)
                
                # Epistemic uncertainty increases with distance from training data
                epistemic_variance = stats['std']**2 * (1 + 0.1 * normalized_distance)
                epistemic_std = np.sqrt(epistemic_variance)
                
                # Confidence interval
                z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
                ci_lower = value - z_score * epistemic_std
                ci_upper = value + z_score * epistemic_std
                
                uncertainty = UncertaintyEstimate(
                    mean=value,
                    variance=epistemic_variance,
                    std=epistemic_std,
                    confidence_interval=(ci_lower, ci_upper),
                    uncertainty_type=UncertaintyType.EPISTEMIC,
                    confidence_level=self.confidence_level
                )
            
            uncertainties[feature_name] = uncertainty
        
        return uncertainties
    
    def estimate_aleatoric_uncertainty(self, 
                                     feature_values: Dict[str, float],
                                     measurement_noise: Optional[Dict[str, float]] = None) -> Dict[str, UncertaintyEstimate]:
        """
        Estimate aleatoric uncertainty (data uncertainty)
        
        Args:
            feature_values: Dictionary of physics feature values
            measurement_noise: Optional measurement noise estimates
            
        Returns:
            Dictionary of uncertainty estimates for each feature
        """
        uncertainties = {}
        
        # Default measurement noise levels for physics features
        default_noise = {
            'delta_fr': 0.1,  # 10% relative noise
            'delta_ft': 0.005,  # Absolute noise
            'delta_fv': 0.05   # Absolute noise
        }
        
        for feature_name, value in feature_values.items():
            # Get noise level
            if measurement_noise and feature_name in measurement_noise:
                noise_level = measurement_noise[feature_name]
            elif feature_name in default_noise:
                noise_level = default_noise[feature_name]
            else:
                # Default to 5% relative noise
                noise_level = abs(value) * 0.05
            
            # Aleatoric uncertainty is primarily from measurement noise
            aleatoric_variance = noise_level**2
            aleatoric_std = noise_level
            
            # Confidence interval
            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
            ci_lower = value - z_score * aleatoric_std
            ci_upper = value + z_score * aleatoric_std
            
            uncertainty = UncertaintyEstimate(
                mean=value,
                variance=aleatoric_variance,
                std=aleatoric_std,
                confidence_interval=(ci_lower, ci_upper),
                uncertainty_type=UncertaintyType.ALEATORIC,
                confidence_level=self.confidence_level
            )
            
            uncertainties[feature_name] = uncertainty
        
        return uncertainties
    
    def estimate_total_uncertainty(self, 
                                 feature_values: Dict[str, float],
                                 epistemic_uncertainties: Optional[Dict[str, UncertaintyEstimate]] = None,
                                 aleatoric_uncertainties: Optional[Dict[str, UncertaintyEstimate]] = None) -> Dict[str, UncertaintyEstimate]:
        """
        Estimate total uncertainty (epistemic + aleatoric)
        
        Args:
            feature_values: Dictionary of physics feature values
            epistemic_uncertainties: Pre-computed epistemic uncertainties
            aleatoric_uncertainties: Pre-computed aleatoric uncertainties
            
        Returns:
            Dictionary of total uncertainty estimates
        """
        # Compute individual uncertainties if not provided
        if epistemic_uncertainties is None:
            epistemic_uncertainties = self.estimate_epistemic_uncertainty(feature_values)
        
        if aleatoric_uncertainties is None:
            aleatoric_uncertainties = self.estimate_aleatoric_uncertainty(feature_values)
        
        total_uncertainties = {}
        
        for feature_name, value in feature_values.items():
            epistemic = epistemic_uncertainties.get(feature_name)
            aleatoric = aleatoric_uncertainties.get(feature_name)
            
            if epistemic is None or aleatoric is None:
                # Fallback to default uncertainty
                total_uncertainties[feature_name] = self._default_total_uncertainty(value)
                continue
            
            # Combine uncertainties (variances add)
            total_variance = epistemic.variance + aleatoric.variance
            total_std = np.sqrt(total_variance)
            
            # Confidence interval for total uncertainty
            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
            ci_lower = value - z_score * total_std
            ci_upper = value + z_score * total_std
            
            uncertainty = UncertaintyEstimate(
                mean=value,
                variance=total_variance,
                std=total_std,
                confidence_interval=(ci_lower, ci_upper),
                uncertainty_type=UncertaintyType.TOTAL,
                confidence_level=self.confidence_level
            )
            
            total_uncertainties[feature_name] = uncertainty
        
        return total_uncertainties
    
    def bootstrap_uncertainty(self, 
                            feature_samples: Dict[str, List[float]],
                            statistic_func: callable = np.mean) -> Dict[str, UncertaintyEstimate]:
        """
        Estimate uncertainty using bootstrap resampling
        
        Args:
            feature_samples: Dictionary of feature sample lists
            statistic_func: Function to compute statistic (default: mean)
            
        Returns:
            Dictionary of bootstrap uncertainty estimates
        """
        uncertainties = {}
        
        for feature_name, samples in feature_samples.items():
            if len(samples) < 2:
                # Not enough samples for bootstrap
                uncertainties[feature_name] = self._default_total_uncertainty(samples[0] if samples else 0.0)
                continue
            
            samples_array = np.array(samples)
            bootstrap_stats = []
            
            # Perform bootstrap resampling
            for _ in range(self.bootstrap_samples):
                # Resample with replacement
                bootstrap_sample = np.random.choice(samples_array, size=len(samples_array), replace=True)
                bootstrap_stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)
            
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Calculate statistics
            mean_stat = np.mean(bootstrap_stats)
            variance_stat = np.var(bootstrap_stats)
            std_stat = np.std(bootstrap_stats)
            
            # Confidence interval from bootstrap distribution
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            
            uncertainty = UncertaintyEstimate(
                mean=mean_stat,
                variance=variance_stat,
                std=std_stat,
                confidence_interval=(ci_lower, ci_upper),
                uncertainty_type=UncertaintyType.TOTAL,
                confidence_level=self.confidence_level
            )
            
            uncertainties[feature_name] = uncertainty
        
        return uncertainties
    
    def calibrate_confidence(self, 
                           predicted_probabilities: np.ndarray,
                           true_labels: np.ndarray,
                           method: str = "platt") -> Dict[str, Any]:
        """
        Calibrate confidence scores using historical data
        
        Args:
            predicted_probabilities: Array of predicted probabilities
            true_labels: Array of true binary labels
            method: Calibration method ("platt" or "isotonic")
            
        Returns:
            Dictionary with calibration results
        """
        if not self.enable_calibration:
            return {'calibrated': False, 'method': None}
        
        try:
            from sklearn.calibration import calibration_curve, CalibratedClassifierCV
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
            
            # Compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                true_labels, predicted_probabilities, n_bins=10
            )
            
            # Calculate calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Fit calibration model
            if method == "platt":
                # Platt scaling (logistic regression)
                calibrator = LogisticRegression()
                calibrator.fit(predicted_probabilities.reshape(-1, 1), true_labels)
                calibration_params = {
                    'method': 'platt',
                    'coef': calibrator.coef_[0][0],
                    'intercept': calibrator.intercept_[0]
                }
            elif method == "isotonic":
                # Isotonic regression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(predicted_probabilities, true_labels)
                calibration_params = {
                    'method': 'isotonic',
                    'calibrator': calibrator
                }
            else:
                raise ValueError(f"Unknown calibration method: {method}")
            
            return {
                'calibrated': True,
                'method': method,
                'calibration_error': calibration_error,
                'calibration_params': calibration_params,
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
            
        except ImportError:
            logger.warning("sklearn not available for confidence calibration")
            return {'calibrated': False, 'method': None, 'error': 'sklearn_not_available'}
        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
            return {'calibrated': False, 'method': None, 'error': str(e)}
    
    def propagate_uncertainty(self, 
                            input_uncertainties: Dict[str, UncertaintyEstimate],
                            function: callable,
                            method: str = "monte_carlo") -> UncertaintyEstimate:
        """
        Propagate uncertainty through a function
        
        Args:
            input_uncertainties: Dictionary of input uncertainties
            function: Function to propagate uncertainty through
            method: Propagation method ("monte_carlo" or "linear")
            
        Returns:
            Output uncertainty estimate
        """
        if method == "monte_carlo":
            return self._monte_carlo_propagation(input_uncertainties, function)
        elif method == "linear":
            return self._linear_propagation(input_uncertainties, function)
        else:
            raise ValueError(f"Unknown propagation method: {method}")
    
    def _monte_carlo_propagation(self, 
                               input_uncertainties: Dict[str, UncertaintyEstimate],
                               function: callable) -> UncertaintyEstimate:
        """Monte Carlo uncertainty propagation"""
        n_samples = 1000
        output_samples = []
        
        for _ in range(n_samples):
            # Sample from input distributions
            sample_inputs = {}
            for name, uncertainty in input_uncertainties.items():
                # Assume normal distribution
                sample_value = np.random.normal(uncertainty.mean, uncertainty.std)
                sample_inputs[name] = sample_value
            
            # Evaluate function
            try:
                output_value = function(sample_inputs)
                output_samples.append(output_value)
            except Exception:
                # Skip failed evaluations
                continue
        
        if not output_samples:
            # Fallback if all evaluations failed
            return self._default_total_uncertainty(0.0)
        
        output_samples = np.array(output_samples)
        
        # Calculate output statistics
        mean_output = np.mean(output_samples)
        variance_output = np.var(output_samples)
        std_output = np.std(output_samples)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(output_samples, 100 * alpha / 2)
        ci_upper = np.percentile(output_samples, 100 * (1 - alpha / 2))
        
        return UncertaintyEstimate(
            mean=mean_output,
            variance=variance_output,
            std=std_output,
            confidence_interval=(ci_lower, ci_upper),
            uncertainty_type=UncertaintyType.TOTAL,
            confidence_level=self.confidence_level
        )
    
    def _linear_propagation(self, 
                          input_uncertainties: Dict[str, UncertaintyEstimate],
                          function: callable) -> UncertaintyEstimate:
        """Linear uncertainty propagation (first-order Taylor expansion)"""
        # This is a simplified implementation
        # In practice, would need to compute gradients numerically or analytically
        
        # For now, use a simple approximation
        input_means = {name: unc.mean for name, unc in input_uncertainties.items()}
        mean_output = function(input_means)
        
        # Approximate variance using sum of input variances (assumes independence)
        total_variance = sum(unc.variance for unc in input_uncertainties.values())
        std_output = np.sqrt(total_variance)
        
        # Confidence interval
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = mean_output - z_score * std_output
        ci_upper = mean_output + z_score * std_output
        
        return UncertaintyEstimate(
            mean=mean_output,
            variance=total_variance,
            std=std_output,
            confidence_interval=(ci_lower, ci_upper),
            uncertainty_type=UncertaintyType.TOTAL,
            confidence_level=self.confidence_level
        )
    
    def _default_epistemic_uncertainty(self, value: float) -> UncertaintyEstimate:
        """Default epistemic uncertainty for unknown features"""
        # Use 10% relative uncertainty as default
        std = abs(value) * 0.1
        variance = std**2
        
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = value - z_score * std
        ci_upper = value + z_score * std
        
        return UncertaintyEstimate(
            mean=value,
            variance=variance,
            std=std,
            confidence_interval=(ci_lower, ci_upper),
            uncertainty_type=UncertaintyType.EPISTEMIC,
            confidence_level=self.confidence_level
        )
    
    def _default_total_uncertainty(self, value: float) -> UncertaintyEstimate:
        """Default total uncertainty for fallback cases"""
        # Use 15% relative uncertainty as default
        std = abs(value) * 0.15
        variance = std**2
        
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = value - z_score * std
        ci_upper = value + z_score * std
        
        return UncertaintyEstimate(
            mean=value,
            variance=variance,
            std=std,
            confidence_interval=(ci_lower, ci_upper),
            uncertainty_type=UncertaintyType.TOTAL,
            confidence_level=self.confidence_level
        )
    
    def update_statistics(self, feature_values: Dict[str, float]) -> None:
        """Update feature statistics with new observations"""
        for feature_name, value in feature_values.items():
            if feature_name not in self.feature_statistics:
                # Initialize new feature
                self.feature_statistics[feature_name] = {
                    'mean': value,
                    'std': abs(value) * 0.1,  # Initial guess
                    'samples': [value]
                }
            else:
                # Update existing feature statistics
                stats = self.feature_statistics[feature_name]
                stats['samples'].append(value)
                
                # Keep only recent samples (last 1000)
                if len(stats['samples']) > 1000:
                    stats['samples'] = stats['samples'][-1000:]
                
                # Update mean and std
                samples_array = np.array(stats['samples'])
                stats['mean'] = np.mean(samples_array)
                stats['std'] = np.std(samples_array)
    
    def get_uncertainty_summary(self, uncertainties: Dict[str, UncertaintyEstimate]) -> Dict[str, Any]:
        """Get summary statistics for uncertainties"""
        if not uncertainties:
            return {'num_features': 0}
        
        # Calculate summary statistics
        means = [unc.mean for unc in uncertainties.values()]
        stds = [unc.std for unc in uncertainties.values()]
        interval_widths = [unc.interval_width for unc in uncertainties.values()]
        
        return {
            'num_features': len(uncertainties),
            'mean_uncertainty': np.mean(stds),
            'max_uncertainty': np.max(stds),
            'min_uncertainty': np.min(stds),
            'mean_interval_width': np.mean(interval_widths),
            'confidence_level': self.confidence_level,
            'feature_names': list(uncertainties.keys())
        } 