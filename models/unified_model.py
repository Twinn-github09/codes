"""
Unified Visual Backstory Generation System with both Normal and Bayesian Transformers.
This module provides a flexible interface to use either standard or Bayesian models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from enum import Enum
from typing import Dict, Any, Optional

# Import both models
from models.backstory_model import VisualBackstoryGenerationModel, create_backstory_model_config
from models.bayesian_backstory_model import BayesianVisualBackstoryGenerationModel, create_bayesian_backstory_model_config

class TransformerType(Enum):
    """Enum for different transformer types"""
    NORMAL = "normal"
    BAYESIAN = "bayesian"

class UnifiedVisualBackstoryModel(nn.Module):
    """
    Unified model that can switch between Normal and Bayesian transformers
    """
    
    def __init__(self, config, transformer_type: TransformerType = TransformerType.NORMAL):
        """
        Initialize the unified model
        
        Args:
            config: Configuration object with model parameters
            transformer_type: Type of transformer to use (NORMAL or BAYESIAN)
        """
        super(UnifiedVisualBackstoryModel, self).__init__()
        
        self.transformer_type = transformer_type
        self.config = config
        
        # Add transformer type to config for easy access
        config.transformer_type = transformer_type.value
        
        # Initialize the appropriate model based on transformer type
        if transformer_type == TransformerType.NORMAL:
            self.model = VisualBackstoryGenerationModel(config)
        elif transformer_type == TransformerType.BAYESIAN:
            self.model = BayesianVisualBackstoryGenerationModel(config)
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")
    
    def forward(self, visual_features, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the selected model
        
        Args:
            visual_features: Visual features from the image
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional target labels for training
            
        Returns:
            Dict containing model outputs
        """
        return self.model(visual_features, input_ids, attention_mask, labels)
    
    def generate(self, visual_features, input_ids, **kwargs):
        """
        Generate text using the selected model
        
        Args:
            visual_features: Visual features from the image
            input_ids: Initial input tokens
            **kwargs: Additional generation parameters
            
        Returns:
            Generated sequences and optionally uncertainty scores
        """
        return self.model.generate(visual_features, input_ids, **kwargs)
    
    def get_uncertainty(self, visual_features, input_ids, num_samples=10):
        """
        Get uncertainty estimates (only available for Bayesian model)
        
        Args:
            visual_features: Visual features from the image
            input_ids: Input token IDs
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            Dict with uncertainty estimates or None for normal model
        """
        if self.transformer_type == TransformerType.BAYESIAN:
            return self.model.get_uncertainty(visual_features, input_ids, num_samples)
        else:
            print("Warning: Uncertainty estimation only available for Bayesian model")
            return None
    
    def switch_transformer_type(self, new_type: TransformerType, new_config=None):
        """
        Switch between transformer types (recreates the model)
        
        Args:
            new_type: New transformer type
            new_config: Optional new configuration (uses current if None)
        """
        if new_config is None:
            new_config = self.config
        
        # Save current state if needed
        current_state = self.state_dict() if hasattr(self, 'model') else None
        
        self.transformer_type = new_type
        new_config.transformer_type = new_type.value
        
        # Reinitialize model
        if new_type == TransformerType.NORMAL:
            self.model = VisualBackstoryGenerationModel(new_config)
        elif new_type == TransformerType.BAYESIAN:
            self.model = BayesianVisualBackstoryGenerationModel(new_config)
        
        print(f"Switched to {new_type.value} transformer")
    
    def get_model_info(self):
        """Get information about the current model"""
        return {
            'transformer_type': self.transformer_type.value,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'config': self.config
        }

def create_unified_config(vocab_size, transformer_type: TransformerType = TransformerType.NORMAL, **kwargs):
    """
    Create a unified configuration for either model type
    
    Args:
        vocab_size: Size of the vocabulary
        transformer_type: Type of transformer to configure
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration object
    """
    if transformer_type == TransformerType.NORMAL:
        config = create_backstory_model_config(vocab_size)
    elif transformer_type == TransformerType.BAYESIAN:
        config = create_bayesian_backstory_model_config(vocab_size)
    else:
        raise ValueError(f"Unknown transformer type: {transformer_type}")
    
    # Add any additional configuration parameters
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    # Add transformer type to config
    config.transformer_type = transformer_type.value
    
    return config

def load_unified_model(checkpoint_path: str, transformer_type: TransformerType = None, 
                      vocab_size: int = None, device: str = 'cpu'):
    """
    Load a unified model from checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        transformer_type: Override transformer type (auto-detect if None)
        vocab_size: Vocabulary size (required if not in checkpoint)
        device: Device to load the model on
        
    Returns:
        Loaded UnifiedVisualBackstoryModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configuration from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        if vocab_size is not None:
            config.vocab_size = vocab_size
    else:
        if vocab_size is None:
            raise ValueError("vocab_size must be provided if not in checkpoint")
        # Auto-detect transformer type from checkpoint keys
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        has_bayesian_layers = any('weight_mu' in key or 'weight_rho' in key for key in state_dict.keys())
        detected_type = TransformerType.BAYESIAN if has_bayesian_layers else TransformerType.NORMAL
        
        if transformer_type is None:
            transformer_type = detected_type
        
        config = create_unified_config(vocab_size, transformer_type)
    
    # Override transformer type if specified
    if transformer_type is not None:
        config.transformer_type = transformer_type.value
        actual_type = transformer_type
    else:
        actual_type = TransformerType(config.transformer_type)
    
    # Create and load model
    model = UnifiedVisualBackstoryModel(config, actual_type)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    return model

def create_model_factory():
    """
    Create a factory function for easy model creation
    
    Returns:
        Factory function
    """
    def factory(vocab_size: int, transformer_type: str = "normal", **kwargs):
        """
        Factory function to create models
        
        Args:
            vocab_size: Size of vocabulary
            transformer_type: "normal" or "bayesian"
            **kwargs: Additional config parameters
            
        Returns:
            UnifiedVisualBackstoryModel instance
        """
        t_type = TransformerType(transformer_type.lower())
        config = create_unified_config(vocab_size, t_type, **kwargs)
        return UnifiedVisualBackstoryModel(config, t_type)
    
    return factory

# Global factory instance for easy access
create_model = create_model_factory()

class ModelManager:
    """
    Utility class to manage multiple models and switch between them
    """
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.current_name = None
    
    def add_model(self, name: str, model: UnifiedVisualBackstoryModel):
        """Add a model to the manager"""
        self.models[name] = model
        if self.current_model is None:
            self.current_model = model
            self.current_name = name
    
    def create_and_add_model(self, name: str, vocab_size: int, 
                           transformer_type: str = "normal", **kwargs):
        """Create and add a new model"""
        model = create_model(vocab_size, transformer_type, **kwargs)
        self.add_model(name, model)
        return model
    
    def switch_to(self, name: str):
        """Switch to a different model"""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        
        self.current_model = self.models[name]
        self.current_name = name
        print(f"Switched to model: {name} ({self.current_model.transformer_type.value})")
    
    def get_current_model(self):
        """Get the current active model"""
        return self.current_model
    
    def list_models(self):
        """List all available models"""
        return {name: model.transformer_type.value for name, model in self.models.items()}
    
    def compare_models(self, visual_features, input_ids, **generation_kwargs):
        """
        Compare outputs from all models
        
        Args:
            visual_features: Visual features
            input_ids: Input token IDs
            **generation_kwargs: Generation parameters
            
        Returns:
            Dict with outputs from all models
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"Generating with {name} ({model.transformer_type.value})...")
            
            # Generate with current model
            if model.transformer_type == TransformerType.BAYESIAN:
                output = model.generate(
                    visual_features, input_ids, 
                    output_uncertainty=True, **generation_kwargs
                )
                if isinstance(output, tuple):
                    generated_ids, uncertainty = output
                    results[name] = {
                        'generated_ids': generated_ids,
                        'uncertainty': uncertainty,
                        'type': 'bayesian'
                    }
                else:
                    results[name] = {
                        'generated_ids': output,
                        'type': 'bayesian'
                    }
            else:
                generated_ids = model.generate(visual_features, input_ids, **generation_kwargs)
                results[name] = {
                    'generated_ids': generated_ids,
                    'type': 'normal'
                }
        
        return results

if __name__ == "__main__":
    # Example usage
    vocab_size = 50000
    
    # Create different models
    print("Creating models...")
    
    # Method 1: Direct creation
    normal_model = create_model(vocab_size, "normal")
    bayesian_model = create_model(vocab_size, "bayesian")
    
    print(f"Normal model: {normal_model.get_model_info()['transformer_type']}")
    print(f"Bayesian model: {bayesian_model.get_model_info()['transformer_type']}")
    
    # Method 2: Using model manager
    manager = ModelManager()
    manager.create_and_add_model("standard", vocab_size, "normal")
    manager.create_and_add_model("uncertain", vocab_size, "bayesian")
    
    print(f"Available models: {manager.list_models()}")
    
    # Switch between models
    manager.switch_to("uncertain")
    current = manager.get_current_model()
    print(f"Current model type: {current.transformer_type.value}")
