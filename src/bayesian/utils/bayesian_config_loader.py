"""
Bayesian Configuration Loader
Handles loading and managing Bayesian network configurations
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging
from ..core.bayesian_engine import BayesianConfig

class BayesianConfigManager:
    """Manages Bayesian network configurations"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Default config directory
        if config_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            config_dir = project_root / "config" / "bayesian"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configurations
        self.default_config = self._create_default_config()
        
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default Bayesian configuration"""
        return {
            'bayesian_engine': {
                'enable_temporal_modeling': True,
                'enable_hierarchical_modeling': True,
                'enable_causal_analysis': True,
                'inference_method': 'variational',
                'temporal_window_size': 10,
                'max_inference_time': 5.0,
                'uncertainty_threshold': 0.1
            },
            'temporal_modeling': {
                'window_size': 10,
                'max_history': 20,
                'consistency_thresholds': {
                    'delta_fr_revised': 0.5,
                    'delta_ft_revised': 0.02,
                    'delta_fv_revised': 0.3
                },
                'transition_probabilities': {
                    'same_state': 0.7,
                    'adjacent_state': 0.25,
                    'distant_state': 0.05
                }
            },
            'hierarchical_modeling': {
                'user_weight': 0.3,
                'session_weight': 0.4,
                'audio_weight': 0.3,
                'adaptation_rate': 0.1,
                'min_samples_for_adaptation': 5,
                'user_risk_thresholds': {
                    'low_risk': 0.8,
                    'medium_risk': 0.4
                }
            },
            'causal_analysis': {
                'enable_interventions': True,
                'enable_counterfactuals': True,
                'causal_effects': {
                    'synthesis_algorithm': {
                        'delta_fr_revised': 0.8,
                        'delta_ft_revised': 0.3,
                        'delta_fv_revised': 0.4,
                        'authenticity': -0.9
                    },
                    'algorithm_artifacts': {
                        'delta_fr_revised': 0.7,
                        'delta_ft_revised': 0.2,
                        'delta_fv_revised': 0.3,
                        'authenticity': -0.8
                    }
                }
            },
            'inference': {
                'variational': {
                    'max_iterations': 1000,
                    'tolerance': 1e-6,
                    'learning_rate': 0.01,
                    'use_gpu': False,
                    'structured': False
                },
                'mcmc': {
                    'num_samples': 1000,
                    'burn_in': 100,
                    'thin': 1,
                    'chains': 4
                }
            },
            'discretization': {
                'delta_fr_thresholds': [6.5, 7.5],
                'delta_ft_thresholds': [0.06, 0.08],
                'delta_fv_thresholds': [1.0, 1.5],
                'confidence_thresholds': [0.7, 0.85, 0.95]
            },
            'performance': {
                'enable_gpu_acceleration': True,
                'batch_inference': True,
                'cache_inference_results': True,
                'parallel_user_processing': True,
                'max_concurrent_inferences': 4
            },
            'validation': {
                'cross_validation_folds': 5,
                'test_split_ratio': 0.2,
                'enable_model_validation': True
            }
        }
    
    def load_config(self, config_name: str = "default") -> BayesianConfig:
        """
        Load Bayesian configuration from file
        
        Args:
            config_name: Name of configuration to load
            
        Returns:
            BayesianConfig object
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                self.logger.info(f"Loaded Bayesian config from {config_file}")
                return self._dict_to_bayesian_config(config_data)
            
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_file}: {e}")
                return self._dict_to_bayesian_config(self.default_config)
        else:
            # Create default config file
            self.save_config(self.default_config, config_name)
            return self._dict_to_bayesian_config(self.default_config)
    
    def save_config(self, config: Dict[str, Any], config_name: str = "default"):
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            config_name: Name for the configuration
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Saved Bayesian config to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {config_file}: {e}")
    
    def _dict_to_bayesian_config(self, config_dict: Dict[str, Any]) -> BayesianConfig:
        """Convert dictionary to BayesianConfig object"""
        engine_config = config_dict.get('bayesian_engine', {})
        
        return BayesianConfig(
            enable_temporal_modeling=engine_config.get('enable_temporal_modeling', True),
            enable_hierarchical_modeling=engine_config.get('enable_hierarchical_modeling', True),
            enable_causal_analysis=engine_config.get('enable_causal_analysis', True),
            inference_method=engine_config.get('inference_method', 'variational'),
            temporal_window_size=engine_config.get('temporal_window_size', 10),
            max_inference_time=engine_config.get('max_inference_time', 5.0),
            uncertainty_threshold=engine_config.get('uncertainty_threshold', 0.1)
        )
    
    def get_full_config(self, config_name: str = "default") -> Dict[str, Any]:
        """Get full configuration dictionary"""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"Failed to load full config: {e}")
                return self.default_config
        else:
            return self.default_config
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """
        Update existing configuration with new values
        
        Args:
            config_name: Name of configuration to update
            updates: Dictionary of updates to apply
        """
        current_config = self.get_full_config(config_name)
        
        # Deep merge updates
        self._deep_merge(current_config, updates)
        
        # Save updated config
        self.save_config(current_config, config_name)
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep merge update dictionary into base dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def create_profile_config(self, 
                            profile_name: str,
                            base_config: str = "default",
                            custom_settings: Optional[Dict[str, Any]] = None) -> BayesianConfig:
        """
        Create a configuration profile for specific use cases
        
        Args:
            profile_name: Name for the new profile
            base_config: Base configuration to start from
            custom_settings: Custom settings to apply
            
        Returns:
            BayesianConfig object
        """
        base_config_dict = self.get_full_config(base_config)
        
        if custom_settings:
            self._deep_merge(base_config_dict, custom_settings)
        
        # Apply profile-specific optimizations
        if profile_name == "real_time":
            # Optimize for real-time processing
            real_time_settings = {
                'bayesian_engine': {
                    'max_inference_time': 1.0,
                    'temporal_window_size': 5
                },
                'inference': {
                    'variational': {
                        'max_iterations': 100,
                        'tolerance': 1e-4
                    }
                },
                'performance': {
                    'enable_gpu_acceleration': True,
                    'batch_inference': False,
                    'max_concurrent_inferences': 1
                }
            }
            self._deep_merge(base_config_dict, real_time_settings)
            
        elif profile_name == "high_accuracy":
            # Optimize for highest accuracy
            accuracy_settings = {
                'bayesian_engine': {
                    'max_inference_time': 10.0,
                    'temporal_window_size': 20,
                    'uncertainty_threshold': 0.05
                },
                'inference': {
                    'variational': {
                        'max_iterations': 2000,
                        'tolerance': 1e-8,
                        'structured': True
                    }
                },
                'temporal_modeling': {
                    'max_history': 50
                }
            }
            self._deep_merge(base_config_dict, accuracy_settings)
            
        elif profile_name == "lightweight":
            # Minimize resource usage
            lightweight_settings = {
                'bayesian_engine': {
                    'enable_temporal_modeling': False,
                    'enable_hierarchical_modeling': False,
                    'enable_causal_analysis': False,
                    'max_inference_time': 0.5
                },
                'performance': {
                    'enable_gpu_acceleration': False,
                    'batch_inference': False,
                    'cache_inference_results': False,
                    'max_concurrent_inferences': 1
                }
            }
            self._deep_merge(base_config_dict, lightweight_settings)
        
        # Save profile config
        self.save_config(base_config_dict, profile_name)
        
        return self._dict_to_bayesian_config(base_config_dict)
    
    def validate_config(self, config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration dictionary
        
        Args:
            config_dict: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required sections
        required_sections = ['bayesian_engine', 'temporal_modeling', 'hierarchical_modeling']
        for section in required_sections:
            if section not in config_dict:
                errors.append(f"Missing required section: {section}")
        
        # Validate bayesian_engine section
        if 'bayesian_engine' in config_dict:
            engine_config = config_dict['bayesian_engine']
            
            # Check inference method
            valid_methods = ['variational', 'mcmc', 'exact']
            inference_method = engine_config.get('inference_method', 'variational')
            if inference_method not in valid_methods:
                errors.append(f"Invalid inference method: {inference_method}. Must be one of {valid_methods}")
            
            # Check temporal window size
            window_size = engine_config.get('temporal_window_size', 10)
            if not isinstance(window_size, int) or window_size < 1:
                errors.append("temporal_window_size must be a positive integer")
            
            # Check max inference time
            max_time = engine_config.get('max_inference_time', 5.0)
            if not isinstance(max_time, (int, float)) or max_time <= 0:
                errors.append("max_inference_time must be a positive number")
        
        # Validate thresholds
        if 'discretization' in config_dict:
            disc_config = config_dict['discretization']
            
            for threshold_name in ['delta_fr_thresholds', 'delta_ft_thresholds', 'delta_fv_thresholds']:
                if threshold_name in disc_config:
                    thresholds = disc_config[threshold_name]
                    if not isinstance(thresholds, list) or len(thresholds) != 2:
                        errors.append(f"{threshold_name} must be a list of 2 values")
                    elif thresholds[0] >= thresholds[1]:
                        errors.append(f"{threshold_name} values must be in ascending order")
        
        return len(errors) == 0, errors
    
    def get_available_configs(self) -> List[str]:
        """Get list of available configuration files"""
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in config_files]

def load_bayesian_config(config_name: str = "default", 
                        config_dir: Optional[str] = None) -> BayesianConfig:
    """
    Convenience function to load Bayesian configuration
    
    Args:
        config_name: Name of configuration to load
        config_dir: Directory containing configuration files
        
    Returns:
        BayesianConfig object
    """
    manager = BayesianConfigManager(config_dir)
    return manager.load_config(config_name) 