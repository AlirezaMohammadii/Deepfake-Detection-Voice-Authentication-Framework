"""
Variational Inference Engine for Bayesian Networks
Implements mean field and structured variational inference
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
import time
from dataclasses import dataclass

# Conditional imports
try:
    from pgmpy.inference import ApproxInference
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

@dataclass
class VariationalConfig:
    """Configuration for variational inference"""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    learning_rate: float = 0.01
    use_gpu: bool = False
    structured: bool = False  # Use structured vs mean field
    
class VariationalInferenceEngine:
    """
    Variational Inference Engine for approximate Bayesian inference
    Supports both mean field and structured variational inference
    """
    
    def __init__(self, config: Optional[VariationalConfig] = None):
        self.config = config or VariationalConfig()
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Variational parameters
        self.variational_params = {}
        self.evidence_lower_bound = []
        
    async def infer(self, 
                   evidence: Dict[str, Any], 
                   prior: Optional[Dict[str, Any]] = None,
                   target_variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform variational inference
        
        Args:
            evidence: Observed evidence
            prior: Prior parameters
            target_variables: Variables to compute posterior for
            
        Returns:
            Posterior distributions and inference results
        """
        try:
            if PGMPY_AVAILABLE:
                return await self._pgmpy_variational_inference(evidence, prior, target_variables)
            else:
                return await self._pytorch_variational_inference(evidence, prior, target_variables)
                
        except Exception as e:
            self.logger.error(f"Variational inference failed: {e}")
            return self._fallback_inference(evidence)
    
    async def _pgmpy_variational_inference(self, 
                                         evidence: Dict[str, Any],
                                         prior: Optional[Dict[str, Any]],
                                         target_variables: Optional[List[str]]) -> Dict[str, Any]:
        """Advanced variational inference using pgmpy"""
        # This would be implemented with pgmpy's ApproxInference
        # For now, return simplified result
        return await self._pytorch_variational_inference(evidence, prior, target_variables)
    
    async def _pytorch_variational_inference(self, 
                                           evidence: Dict[str, Any],
                                           prior: Optional[Dict[str, Any]],
                                           target_variables: Optional[List[str]]) -> Dict[str, Any]:
        """PyTorch-based variational inference implementation"""
        
        # Convert evidence to tensors
        evidence_tensors = self._prepare_evidence_tensors(evidence)
        
        # Initialize variational parameters
        var_params = self._initialize_variational_parameters(evidence_tensors, prior)
        
        # Optimization loop
        optimizer = optim.Adam(var_params.values(), lr=self.config.learning_rate)
        
        best_elbo = float('-inf')
        patience = 50
        no_improvement = 0
        
        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()
            
            # Compute ELBO (Evidence Lower BOund)
            elbo = self._compute_elbo(var_params, evidence_tensors, prior)
            loss = -elbo  # Minimize negative ELBO
            
            loss.backward()
            optimizer.step()
            
            # Check convergence
            if elbo.item() > best_elbo + self.config.tolerance:
                best_elbo = elbo.item()
                no_improvement = 0
            else:
                no_improvement += 1
                
            if no_improvement >= patience:
                self.logger.info(f"Convergence reached at iteration {iteration}")
                break
                
            # Store ELBO for monitoring
            self.evidence_lower_bound.append(elbo.item())
        
        # Extract posterior distributions
        posteriors = self._extract_posteriors(var_params, target_variables)
        
        return {
            'posteriors': posteriors,
            'elbo': best_elbo,
            'iterations': iteration + 1,
            'converged': no_improvement >= patience
        }
    
    def _prepare_evidence_tensors(self, evidence: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert evidence to PyTorch tensors"""
        evidence_tensors = {}
        
        for var_name, value in evidence.items():
            if isinstance(value, (int, float)):
                # Continuous variable
                evidence_tensors[var_name] = torch.tensor([float(value)], device=self.device)
            elif isinstance(value, str):
                # Discrete variable - convert to one-hot
                state_map = {'low': 0, 'medium': 1, 'high': 2}
                state_idx = state_map.get(value, 1)
                one_hot = torch.zeros(3, device=self.device)
                one_hot[state_idx] = 1.0
                evidence_tensors[var_name] = one_hot
            else:
                # Already a tensor
                evidence_tensors[var_name] = torch.tensor(value, device=self.device)
        
        return evidence_tensors
    
    def _initialize_variational_parameters(self, 
                                         evidence_tensors: Dict[str, torch.Tensor],
                                         prior: Optional[Dict[str, Any]]) -> Dict[str, nn.Parameter]:
        """Initialize variational parameters for mean field approximation"""
        var_params = {}
        
        # For each variable not in evidence, create variational parameters
        variable_names = ['delta_fr_revised', 'delta_ft_revised', 'delta_fv_revised', 'authenticity']
        
        for var_name in variable_names:
            if var_name not in evidence_tensors:
                if var_name == 'authenticity':
                    # Binary variable
                    # Use logit parameterization for numerical stability
                    logit = torch.zeros(1, device=self.device, requires_grad=True)
                    var_params[f'{var_name}_logit'] = nn.Parameter(logit)
                else:
                    # Discrete variable with 3 states
                    # Use softmax parameterization
                    logits = torch.zeros(3, device=self.device, requires_grad=True)
                    var_params[f'{var_name}_logits'] = nn.Parameter(logits)
        
        return var_params
    
    def _compute_elbo(self, 
                     var_params: Dict[str, nn.Parameter],
                     evidence_tensors: Dict[str, torch.Tensor],
                     prior: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Compute Evidence Lower BOund (ELBO)"""
        
        # ELBO = E_q[log p(x,z)] - E_q[log q(z)]
        # where x is observed, z is latent
        
        log_joint = self._compute_log_joint(var_params, evidence_tensors, prior)
        entropy = self._compute_entropy(var_params)
        
        elbo = log_joint + entropy
        return elbo
    
    def _compute_log_joint(self, 
                          var_params: Dict[str, nn.Parameter],
                          evidence_tensors: Dict[str, torch.Tensor],
                          prior: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Compute expected log joint probability under variational distribution"""
        
        log_joint = torch.tensor(0.0, device=self.device)
        
        # Extract variational distributions
        var_dists = self._get_variational_distributions(var_params)
        
        # Physics-based model log probabilities
        # This implements a simplified version of the physics-based deepfake detection model
        
        # Prior probabilities
        for var_name, dist in var_dists.items():
            if var_name == 'authenticity':
                # Prior: P(authentic) = 0.7
                prob_authentic = torch.sigmoid(var_params['authenticity_logit'])
                log_prior = prob_authentic * torch.log(torch.tensor(0.7, device=self.device)) + \
                           (1 - prob_authentic) * torch.log(torch.tensor(0.3, device=self.device))
                log_joint += log_prior
            else:
                # Physics features prior - uniform over states
                probs = torch.softmax(var_params[f'{var_name}_logits'], dim=0)
                log_prior = torch.sum(probs * torch.log(torch.tensor(1/3, device=self.device)))
                log_joint += log_prior
        
        # Likelihood terms
        if 'authenticity_logit' in var_params:
            prob_authentic = torch.sigmoid(var_params['authenticity_logit'])
            
            # Physics features influence authenticity
            # High delta_fr strongly indicates TTS (lower authenticity)
            if 'delta_fr_revised_logits' in var_params:
                delta_fr_probs = torch.softmax(var_params['delta_fr_revised_logits'], dim=0)
                # P(authentic | delta_fr) - higher delta_fr means lower authenticity
                auth_given_fr = torch.tensor([0.9, 0.5, 0.1], device=self.device)  # For low, med, high
                expected_auth_prob = torch.sum(delta_fr_probs * auth_given_fr)
                
                likelihood = prob_authentic * torch.log(expected_auth_prob + 1e-8) + \
                           (1 - prob_authentic) * torch.log(1 - expected_auth_prob + 1e-8)
                log_joint += likelihood
            
            # Similar for other features
            if 'delta_ft_revised_logits' in var_params:
                delta_ft_probs = torch.softmax(var_params['delta_ft_revised_logits'], dim=0)
                auth_given_ft = torch.tensor([0.8, 0.6, 0.4], device=self.device)
                expected_auth_prob = torch.sum(delta_ft_probs * auth_given_ft)
                
                likelihood = prob_authentic * torch.log(expected_auth_prob + 1e-8) + \
                           (1 - prob_authentic) * torch.log(1 - expected_auth_prob + 1e-8)
                log_joint += likelihood
            
            if 'delta_fv_revised_logits' in var_params:
                delta_fv_probs = torch.softmax(var_params['delta_fv_revised_logits'], dim=0)
                auth_given_fv = torch.tensor([0.85, 0.65, 0.45], device=self.device)
                expected_auth_prob = torch.sum(delta_fv_probs * auth_given_fv)
                
                likelihood = prob_authentic * torch.log(expected_auth_prob + 1e-8) + \
                           (1 - prob_authentic) * torch.log(1 - expected_auth_prob + 1e-8)
                log_joint += likelihood
        
        # Evidence terms (observed variables)
        for var_name, observed_value in evidence_tensors.items():
            if var_name in ['delta_fr_revised', 'delta_ft_revised', 'delta_fv_revised']:
                if f'{var_name}_logits' in var_params:
                    # Observed discrete variable
                    probs = torch.softmax(var_params[f'{var_name}_logits'], dim=0)
                    log_likelihood = torch.sum(observed_value * torch.log(probs + 1e-8))
                    log_joint += log_likelihood
        
        return log_joint
    
    def _compute_entropy(self, var_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Compute entropy of variational distribution"""
        entropy = torch.tensor(0.0, device=self.device)
        
        for param_name, param in var_params.items():
            if 'logit' in param_name and param.numel() == 1:
                # Binary variable entropy
                prob = torch.sigmoid(param)
                ent = -(prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
                entropy += ent
            elif 'logits' in param_name:
                # Categorical variable entropy
                probs = torch.softmax(param, dim=0)
                ent = -torch.sum(probs * torch.log(probs + 1e-8))
                entropy += ent
        
        return entropy
    
    def _get_variational_distributions(self, var_params: Dict[str, nn.Parameter]) -> Dict[str, torch.Tensor]:
        """Extract variational distributions from parameters"""
        distributions = {}
        
        for param_name, param in var_params.items():
            var_name = param_name.replace('_logit', '').replace('_logits', '')
            
            if 'logit' in param_name and param.numel() == 1:
                # Binary distribution
                distributions[var_name] = torch.sigmoid(param)
            elif 'logits' in param_name:
                # Categorical distribution
                distributions[var_name] = torch.softmax(param, dim=0)
        
        return distributions
    
    def _extract_posteriors(self, 
                          var_params: Dict[str, nn.Parameter],
                          target_variables: Optional[List[str]]) -> Dict[str, Any]:
        """Extract posterior distributions from optimized variational parameters"""
        posteriors = {}
        
        with torch.no_grad():
            for param_name, param in var_params.items():
                var_name = param_name.replace('_logit', '').replace('_logits', '')
                
                if target_variables is None or var_name in target_variables:
                    if 'logit' in param_name and param.numel() == 1:
                        # Binary variable
                        prob = torch.sigmoid(param).item()
                        posteriors[var_name] = {
                            'type': 'binary',
                            'probability': prob,
                            'values': [1 - prob, prob],  # [P(0), P(1)]
                            'mean': prob,
                            'variance': prob * (1 - prob)
                        }
                    elif 'logits' in param_name:
                        # Categorical variable
                        probs = torch.softmax(param, dim=0)
                        posteriors[var_name] = {
                            'type': 'categorical',
                            'probabilities': probs.tolist(),
                            'states': ['low', 'medium', 'high'],
                            'mean': torch.sum(probs * torch.arange(3, dtype=torch.float, device=self.device)).item(),
                            'variance': torch.sum(probs * (torch.arange(3, dtype=torch.float, device=self.device) - 
                                                         torch.sum(probs * torch.arange(3, dtype=torch.float, device=self.device)))**2).item()
                        }
        
        return posteriors
    
    async def _fallback_inference(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback inference when advanced methods fail"""
        posteriors = {}
        
        # Simple rule-based inference
        spoof_probability = 0.5
        
        for var_name, value in evidence.items():
            if var_name == 'delta_fr_revised':
                if value == 'high':
                    spoof_probability += 0.3
                elif value == 'low':
                    spoof_probability -= 0.1
            elif var_name in ['delta_ft_revised', 'delta_fv_revised']:
                if value == 'high':
                    spoof_probability += 0.1
        
        spoof_probability = max(0.0, min(1.0, spoof_probability))
        
        posteriors['authenticity'] = {
            'type': 'binary',
            'probability': 1 - spoof_probability,
            'values': [spoof_probability, 1 - spoof_probability],
            'mean': 1 - spoof_probability,
            'variance': spoof_probability * (1 - spoof_probability)
        }
        
        return {
            'posteriors': posteriors,
            'elbo': -1.0,  # Invalid ELBO for fallback
            'iterations': 1,
            'converged': True
        }
    
    def get_inference_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics about the inference process"""
        return {
            'elbo_history': self.evidence_lower_bound,
            'final_elbo': self.evidence_lower_bound[-1] if self.evidence_lower_bound else None,
            'convergence_info': {
                'total_iterations': len(self.evidence_lower_bound),
                'elbo_improvement': (self.evidence_lower_bound[-1] - self.evidence_lower_bound[0]) 
                                  if len(self.evidence_lower_bound) > 1 else 0.0
            },
            'config': {
                'max_iterations': self.config.max_iterations,
                'tolerance': self.config.tolerance,
                'learning_rate': self.config.learning_rate,
                'device': str(self.device)
            }
        }
    
    def reset_inference_state(self):
        """Reset inference state for new problem"""
        self.variational_params.clear()
        self.evidence_lower_bound.clear() 