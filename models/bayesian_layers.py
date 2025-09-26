"""
Bayesian Neural Network layers for uncertainty estimation.
These layers implement weight distributions instead of point estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class BayesianLinear(nn.Module):
    """
    Bayesian Linear layer that uses weight distributions instead of point estimates.
    This implements the local reparameterization trick for efficient training.
    """
    
    def __init__(self, in_features, out_features, bias=True, prior_sigma_1=1.0, prior_sigma_2=0.0025, 
                 prior_pi=0.5, posterior_mu_init=0, posterior_rho_init=-3.0):
        """
        Initialize Bayesian Linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias
            prior_sigma_1: Standard deviation of first prior component
            prior_sigma_2: Standard deviation of second prior component 
            prior_pi: Mixture coefficient for the prior mixture
            posterior_mu_init: Initial value for the posterior mean
            posterior_rho_init: Initial value for the posterior rho parameter
        """
        super(BayesianLinear, self).__init__()
        
        # Layer dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior distribution parameters
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Weight parameters (mean and rho for the posterior)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * posterior_rho_init)
        
        # Bias parameters if needed
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.ones(out_features) * posterior_rho_init)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            
        self.bias = bias
        
        # Initialize log_prior and log_variational_posterior
        self.log_prior = 0
        self.log_variational_posterior = 0
        
    def forward(self, x, return_kl=True):
        """
        Forward pass through the Bayesian layer with the local reparameterization trick
        
        Args:
            x: Input tensor
            return_kl: Whether to return the KL divergence
            
        Returns:
            Output tensor after Bayesian linear transformation
        """
        # Calculate weight standard deviation from rho parameter
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        
        # Local reparameterization trick: sample from output distribution directly
        # Compute activation mean and variance
        act_mu = F.linear(x, self.weight_mu)  # mean of activations
        act_var = F.linear(x.pow(2), weight_sigma.pow(2))  # variance of activations
        
        # Sample from Gaussian with these parameters
        eps = torch.randn_like(act_mu)
        act = act_mu + torch.sqrt(act_var) * eps
        
        # Add bias if needed
        if self.bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_eps * bias_sigma
            act = act + bias
        
        # Calculate KL divergence between posterior and prior
        if return_kl:
            self.log_prior = self.calculate_log_prior()
            self.log_variational_posterior = self.calculate_log_variational_posterior()
            kl = self.log_variational_posterior - self.log_prior
            return act, kl
        
        return act
    
    def calculate_log_prior(self):
        """Calculate log prior for the layer"""
        if self.bias:
            return self._log_scale_mixture_prior(self.weight_mu, self.weight_sigma) + \
                   self._log_scale_mixture_prior(self.bias_mu, self.bias_sigma)
        else:
            return self._log_scale_mixture_prior(self.weight_mu, self.weight_sigma)
    
    def calculate_log_variational_posterior(self):
        """Calculate log posterior for the layer"""
        if self.bias:
            return self._log_gaussian(self.weight_mu, self.weight_sigma) + \
                   self._log_gaussian(self.bias_mu, self.bias_sigma)
        else:
            return self._log_gaussian(self.weight_mu, self.weight_sigma)
    
    def _log_scale_mixture_prior(self, mu, sigma):
        """Log probability under scale mixture prior"""
        sigma_squared = sigma.pow(2)
        log_mix1 = -0.5 * torch.log(2 * math.pi * self.prior_sigma_1**2) - \
                   0.5 * mu.pow(2) / (self.prior_sigma_1**2) - \
                   0.5 * sigma_squared / (self.prior_sigma_1**2)
                   
        log_mix2 = -0.5 * torch.log(2 * math.pi * self.prior_sigma_2**2) - \
                   0.5 * mu.pow(2) / (self.prior_sigma_2**2) - \
                   0.5 * sigma_squared / (self.prior_sigma_2**2)
        
        return torch.log(self.prior_pi * torch.exp(log_mix1) + (1 - self.prior_pi) * torch.exp(log_mix2))
    
    def _log_gaussian(self, mu, sigma):
        """Log probability under Gaussian distribution"""
        return -0.5 * torch.log(2 * math.pi) - torch.log(sigma) - 0.5 * (mu / sigma).pow(2)


class BayesianTransformerDecoderLayer(nn.Module):
    """
    Bayesian Transformer Decoder Layer that uses Bayesian Linear layers instead of standard ones.
    This provides uncertainty estimates in the transformer's predictions.
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="gelu", layer_norm_eps=1e-5, batch_first=False):
        """
        Initialize Bayesian Transformer Decoder Layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function
            layer_norm_eps: Layer normalization epsilon
            batch_first: Whether batch is the first dimension
        """
        super(BayesianTransformerDecoderLayer, self).__init__()
        
        # Self-attention layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # Cross-attention layers
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # Bayesian feed-forward network
        self.bayesian_linear1 = BayesianLinear(d_model, dim_feedforward)
        self.bayesian_linear2 = BayesianLinear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        self.activation = self._get_activation_fn(activation)
        
        # For compatibility with torch API
        self.batch_first = batch_first
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the Bayesian transformer decoder layer.
        
        Args:
            tgt: Target sequence [batch_size, target_len, d_model] if batch_first
            memory: Memory from encoder [batch_size, memory_len, d_model] if batch_first
            tgt_mask: Target mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
        
        Returns:
            Transformed output and KL divergence
        """
        # Self-attention block
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross-attention block
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask,
                                     key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        
        # Bayesian feed-forward block
        tgt2 = self.norm3(tgt)
        
        # Forward through Bayesian layers with KL calculation
        tgt2, kl_linear1 = self.bayesian_linear1(tgt2, return_kl=True)
        tgt2 = self.activation(tgt2)
        tgt2 = self.dropout(tgt2)
        tgt2, kl_linear2 = self.bayesian_linear2(tgt2, return_kl=True)
        
        # Residual connection and total KL
        tgt = tgt + self.dropout3(tgt2)
        kl = kl_linear1 + kl_linear2
        
        return tgt, kl
    
    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Activation {activation} not supported")


class BayesianTransformerDecoder(nn.Module):
    """
    Bayesian Transformer Decoder that stacks multiple Bayesian Transformer Decoder Layers
    and tracks the total KL divergence across all layers.
    """
    
    def __init__(self, decoder_layer, num_layers, norm=None):
        """
        Initialize Bayesian Transformer Decoder.
        
        Args:
            decoder_layer: Bayesian decoder layer
            num_layers: Number of decoder layers
            norm: Normalization layer
        """
        super(BayesianTransformerDecoder, self).__init__()
        
        # Create a ModuleList of decoder layers
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass through the Bayesian transformer decoder.
        
        Args:
            tgt: Target sequence
            memory: Memory from encoder
            tgt_mask: Target mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
        
        Returns:
            Transformed output and total KL divergence
        """
        output = tgt
        total_kl = 0.0
        
        # Pass through each layer
        for layer in self.layers:
            output, kl = layer(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
            total_kl += kl
        
        # Apply normalization if provided
        if self.norm is not None:
            output = self.norm(output)
        
        return output, total_kl


def create_bayesian_decoder_layer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                              activation="gelu", layer_norm_eps=1e-5, batch_first=True):
    """
    Create a Bayesian transformer decoder layer.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        activation: Activation function
        layer_norm_eps: Layer normalization epsilon
        batch_first: Whether batch is the first dimension
        
    Returns:
        Bayesian transformer decoder layer
    """
    return BayesianTransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        batch_first=batch_first
    )
