"""
Bayesian Visual Backstory Generation Model with CVAE and Bayesian Transformer Decoder.
This module extends the standard model with Bayesian components for uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import GPT2Config

# Import from Visual COMET codebase
from models.detector_feature import SimpleDetector
from models.pytorch_misc import pack_sequence, pad_sequence
from models.bayesian_layers import BayesianTransformerDecoder, create_bayesian_decoder_layer

class BayesianVisualBackstoryGenerationModel(nn.Module):
    """
    Bayesian Visual Backstory Generation Model that combines:
    1. Enhanced visual feature extraction
    2. CVAE for latent representation
    3. Bayesian Transformer decoder for text generation with uncertainty
    """
    
    def __init__(self, config):
        """
        Initialize the model
        
        Args:
            config: Configuration object with model parameters
        """
        super(BayesianVisualBackstoryGenerationModel, self).__init__()
        self.config = config
        
        # Visual feature processing
        self.detector = SimpleDetector(
            pretrained=True, 
            final_dim=config.hidden_dim, 
            use_bbox=config.use_bbox
        )
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # CVAE Encoder
        self.cvae_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2, eps=config.layer_norm_epsilon),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)  # 2x for mean and logvar
        )
        
        # Bayesian Transformer Decoder
        decoder_layer = create_bayesian_decoder_layer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation='gelu',
            layer_norm_eps=config.layer_norm_epsilon,
            batch_first=True
        )
        self.transformer_decoder = BayesianTransformerDecoder(
            decoder_layer, 
            num_layers=config.num_hidden_layers,
            norm=nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        )
        
        # Output projection
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.token_embeddings.weight = self.lm_head.weight
        
        # Initialize weights
        self._init_weights()
        
        # KL weights for both CVAE and Bayesian layers
        self.kl_weight = config.kl_weight
        self.bayesian_kl_weight = config.bayesian_kl_weight if hasattr(config, 'bayesian_kl_weight') else 0.01
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def encode(self, visual_features, text_features=None):
        """
        Encode inputs to latent distribution parameters
        
        Args:
            visual_features: Visual features from the image
            text_features: Optional text features to condition on
            
        Returns:
            Tuple of (mean, logvar) for the latent distribution
        """
        # Process visual features if they're not already processed
        if isinstance(visual_features, dict):
            # Extract object features
            visual_output = self.detector(
                img_feats=visual_features.get('image_features'),
                boxes=visual_features.get('boxes'),
                box_mask=visual_features.get('box_mask'),
                obj_labels=visual_features.get('class_ids')
            )
            visual_features = visual_output['obj_reps']
        
        # Combine visual and text features if available
        if text_features is not None:
            # Average pool text features
            text_features = torch.mean(text_features, dim=1, keepdim=True)
            
            # Concatenate with visual features
            # We use first visual feature which is the whole image
            combined_features = torch.cat([visual_features[:, 0:1, :], text_features], dim=1)
            combined_features = torch.mean(combined_features, dim=1)
        else:
            # Just use the visual features
            combined_features = visual_features[:, 0, :]  # Use whole image feature
        
        # Encode to latent parameters
        latent_params = self.cvae_encoder(combined_features)
        mean, logvar = torch.chunk(latent_params, 2, dim=1)
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: sample from the latent distribution
        
        Args:
            mean: Mean of the latent Gaussian
            logvar: Log variance of the latent Gaussian
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def prepare_decoder_inputs(self, input_ids, attention_mask=None):
        """
        Prepare inputs for the transformer decoder
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Processed inputs with embeddings and position information
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        inputs_embeds = self.token_embeddings(input_ids)
        
        # Create position IDs and embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states, attention_mask
    
    def decode(self, latent_z, input_ids, attention_mask=None):
        """
        Decode from latent space to generate text, tracking Bayesian layer uncertainty
        
        Args:
            latent_z: Sampled latent vector [batch_size, hidden_dim]
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (logits, bayesian_kl) for next token prediction and uncertainty
        """
        # Prepare decoder inputs
        decoder_inputs, attention_mask = self.prepare_decoder_inputs(input_ids, attention_mask)
        
        # Create memory from latent vector
        # Expand latent_z to create a sequence for the encoder output
        batch_size = latent_z.size(0)
        memory = latent_z.unsqueeze(1).expand(-1, 1, -1)  # [batch_size, 1, hidden_dim]
        
        # Create causal mask for decoder
        seq_len = decoder_inputs.shape[1]
        causal_mask = self._generate_causal_mask(seq_len, input_ids.device)
        
        # Process attention mask if provided
        key_padding_mask = None
        if attention_mask is not None:
            # Convert mask: 1 = keep, 0 = mask
            key_padding_mask = (1 - attention_mask).bool()
        
        # Pass through Bayesian transformer decoder
        decoder_outputs, bayesian_kl = self.transformer_decoder(
            tgt=decoder_inputs,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.lm_head(decoder_outputs)
        
        return logits, bayesian_kl
    
    def _generate_causal_mask(self, size, device):
        """Generate a causal attention mask for the transformer decoder"""
        mask = torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, visual_features, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model
        
        Args:
            visual_features: Visual features from the image
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional target labels for training
            
        Returns:
            Dict containing model outputs
        """
        # Encode to latent space
        mean, logvar = self.encode(visual_features)
        
        # Sample from latent space
        latent_z = self.reparameterize(mean, logvar)
        
        # Decode to get token logits and Bayesian layer KL
        logits, bayesian_kl = self.decode(latent_z, input_ids, attention_mask)
        
        outputs = {
            'logits': logits,
            'latent_mean': mean,
            'latent_logvar': logvar,
            'bayesian_kl': bayesian_kl
        }
        
        # Calculate loss if labels are provided
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate reconstruction loss (cross entropy)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            rec_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Calculate KL divergence loss for CVAE
            cvae_kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / mean.size(0)
            
            # Total loss with both KL weights
            total_loss = rec_loss + self.kl_weight * cvae_kl_loss + self.bayesian_kl_weight * bayesian_kl
            
            outputs['loss'] = total_loss
            outputs['rec_loss'] = rec_loss
            outputs['cvae_kl_loss'] = cvae_kl_loss
            outputs['bayesian_kl_loss'] = self.bayesian_kl_weight * bayesian_kl
        
        return outputs
    
    def get_uncertainty(self, visual_features, input_ids, num_samples=10):
        """
        Estimate prediction uncertainty by sampling multiple outputs
        
        Args:
            visual_features: Visual features from the image
            input_ids: Input token IDs
            num_samples: Number of samples to use for uncertainty estimation
            
        Returns:
            Dict with mean logits and uncertainty estimates
        """
        # Encode to latent space
        mean, logvar = self.encode(visual_features)
        
        # Store multiple predictions
        all_logits = []
        
        # Generate multiple samples
        for _ in range(num_samples):
            # Sample from latent space
            latent_z = self.reparameterize(mean, logvar)
            
            # Decode to get token logits
            logits, _ = self.decode(latent_z, input_ids)
            all_logits.append(logits)
        
        # Stack all predictions [num_samples, batch, seq_len, vocab_size]
        stacked_logits = torch.stack(all_logits)
        
        # Calculate mean prediction
        mean_logits = torch.mean(stacked_logits, dim=0)
        
        # Calculate prediction uncertainty (variance across samples)
        uncertainty = torch.var(torch.softmax(stacked_logits, dim=-1), dim=0)
        
        # Take max uncertainty across vocabulary for each position
        token_uncertainty = torch.mean(uncertainty, dim=-1)
        
        return {
            'mean_logits': mean_logits,
            'token_uncertainty': token_uncertainty,
            'all_logits': stacked_logits
        }
    
    @torch.no_grad()
    def generate(self, visual_features, input_ids, max_length=50, temperature=1.0, 
                 top_k=0, top_p=0.9, do_sample=True, num_return_sequences=1, 
                 output_uncertainty=False, num_uncertainty_samples=5):
        """
        Generate text conditioned on visual features with uncertainty estimation
        
        Args:
            visual_features: Visual features from the image
            input_ids: Initial input tokens (usually just BOS token)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to generate
            output_uncertainty: Whether to output token-level uncertainty
            num_uncertainty_samples: Number of samples for uncertainty estimation
            
        Returns:
            Generated token sequences and optionally uncertainty scores
        """
        batch_size = input_ids.shape[0]
        
        # Encode visual features
        mean, logvar = self.encode(visual_features)
        
        # For generation, we can either:
        # 1. Sample from the latent distribution (if do_sample=True)
        # 2. Use the mean vector directly (if do_sample=False)
        if do_sample:
            # Sample multiple latent vectors for diversity
            if num_return_sequences > 1:
                mean = mean.repeat(num_return_sequences, 1)
                logvar = logvar.repeat(num_return_sequences, 1)
                input_ids = input_ids.repeat(num_return_sequences, 1)
            
            latent_z = self.reparameterize(mean, logvar)
        else:
            # Just use the mean for deterministic generation
            if num_return_sequences > 1:
                mean = mean.repeat(num_return_sequences, 1)
                input_ids = input_ids.repeat(num_return_sequences, 1)
            
            latent_z = mean
        
        # Initialize generation
        curr_ids = input_ids.clone()
        
        # For uncertainty estimation
        if output_uncertainty:
            token_uncertainties = []
        
        # Generate tokens auto-regressively
        for i in range(max_length):
            if output_uncertainty:
                # Generate multiple predictions for uncertainty estimation
                all_next_token_logits = []
                for _ in range(num_uncertainty_samples):
                    # Decode with Bayesian network (samples different weights each time)
                    logits, _ = self.decode(latent_z, curr_ids)
                    all_next_token_logits.append(logits[:, -1, :])
                
                # Stack all predictions [num_samples, batch, vocab_size]
                stacked_logits = torch.stack(all_next_token_logits)
                
                # Use mean logits for token selection
                next_token_logits = torch.mean(stacked_logits, dim=0)
                
                # Calculate uncertainty as variance in probabilities
                token_probs = torch.softmax(stacked_logits, dim=-1)
                uncertainty = torch.var(token_probs, dim=0)
                mean_uncertainty = torch.mean(uncertainty, dim=-1)
                token_uncertainties.append(mean_uncertainty)
            else:
                # Get logits for next token (single prediction)
                logits, _ = self.decode(latent_z, curr_ids)
                next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample or argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Check for end of sequence token
            if (next_token == self.config.eos_token_id).all():
                break
        
        if output_uncertainty:
            # Stack all token uncertainties [seq_len, batch_size]
            uncertainty_tensor = torch.stack(token_uncertainties, dim=0)
            return curr_ids, uncertainty_tensor
        
        return curr_ids


def create_bayesian_backstory_model_config(vocab_size):
    """
    Create a configuration for the Bayesian backstory generation model
    
    Args:
        vocab_size: Size of the vocabulary
        
    Returns:
        Configuration object
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=768,  # Hidden dimension
        n_layer=6,  # Number of transformer layers
        n_head=12,  # Number of attention heads
        n_positions=128,  # Maximum sequence length
        bos_token_id=None,  # Will be set by tokenizer
        eos_token_id=None,  # Will be set by tokenizer
        use_bbox=True,  # Use bounding box features
    )
    
    # Add CVAE-specific parameters
    config.kl_weight = 0.1  # Weight for CVAE KL loss
    config.bayesian_kl_weight = 0.01  # Weight for Bayesian layers KL loss
    config.hidden_dim = config.n_embd  # For clarity
    config.intermediate_size = config.hidden_dim * 4
    config.num_hidden_layers = config.n_layer
    config.num_attention_heads = config.n_head
    config.max_position_embeddings = config.n_positions
    config.hidden_dropout_prob = 0.1
    config.layer_norm_epsilon = 1e-5
    
    return config


if __name__ == "__main__":
    # Example usage
    config = create_bayesian_backstory_model_config(vocab_size=50000)
    model = BayesianVisualBackstoryGenerationModel(config)
    
    # Print model architecture
    print(model)
