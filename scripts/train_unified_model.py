"""
Unified Training Script for both Normal and Bayesian Visual Backstory Generation Models.
This script provides a flexible training interface that works with both model types.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

# Import unified model system
from models.unified_model import (
    UnifiedVisualBackstoryModel, TransformerType, 
    create_unified_config, create_model, ModelManager
)
from dataloaders.vcg import VCGDataset, VCGDataLoader, vcg_collate_fn

def train_unified_model(args):
    """
    Train either Normal or Bayesian model based on configuration
    
    Args:
        args: Command-line arguments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training {args.model_type} transformer model")
    
    # Import filtered dataset
    from dataloaders.filtered_dataset import create_filtered_dataloaders
    
    # Create filtered datasets that only include annotations with corresponding features
    train_loader, val_loader = create_filtered_dataloaders(
        annotations_json=args.annotations_json,
        val_annotations_json=args.val_annotations_json,
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=getattr(args, 'num_workers', 4)
    )
    
    # Create model (using standard vocab size for transformers)
    vocab_size = 50257  # Standard GPT-2 vocabulary size
    transformer_type = TransformerType(args.model_type.lower())
    
    # Create configuration with custom parameters
    config_kwargs = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'max_seq_len': args.max_seq_len
    }
    
    # Add model-specific parameters
    if transformer_type == TransformerType.BAYESIAN:
        config_kwargs.update({
            'kl_weight': args.kl_weight,
            'bayesian_kl_weight': args.bayesian_kl_weight,
            'prior_sigma_1': args.prior_sigma_1,
            'prior_sigma_2': args.prior_sigma_2,
            'prior_pi': args.prior_pi
        })
    else:
        config_kwargs['kl_weight'] = args.kl_weight
    
    config = create_unified_config(vocab_size, transformer_type, **config_kwargs)
    model = UnifiedVisualBackstoryModel(config, transformer_type)
    model.to(device)
    
    # Print model information
    model_info = model.get_model_info()
    print(f"Model type: {model_info['transformer_type']}")
    print(f"Total parameters: {model_info['num_parameters']:,}")
    print(f"Trainable parameters: {model_info['num_trainable_parameters']:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_epoch_loss = 0.0
        train_rec_loss = 0.0
        train_kl_loss = 0.0
        
        if transformer_type == TransformerType.BAYESIAN:
            train_bayesian_kl_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch in pbar:
            # Move batch to device
            visual_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch['visual_features'].items()}
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                visual_features=visual_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_epoch_loss += loss.item()
            train_rec_loss += outputs['rec_loss'].item()
            
            if transformer_type == TransformerType.NORMAL:
                train_kl_loss += outputs['kl_loss'].item()
                pbar.set_postfix({
                    'loss': loss.item(),
                    'rec_loss': outputs['rec_loss'].item(),
                    'kl_loss': outputs['kl_loss'].item()
                })
            else:  # Bayesian
                train_kl_loss += outputs['cvae_kl_loss'].item()
                train_bayesian_kl_loss += outputs['bayesian_kl_loss'].item()
                pbar.set_postfix({
                    'loss': loss.item(),
                    'rec_loss': outputs['rec_loss'].item(),
                    'cvae_kl': outputs['cvae_kl_loss'].item(),
                    'bayes_kl': outputs['bayesian_kl_loss'].item()
                })
        
        # Calculate average losses
        num_batches = len(train_loader)
        avg_train_loss = train_epoch_loss / num_batches
        avg_train_rec_loss = train_rec_loss / num_batches
        avg_train_kl_loss = train_kl_loss / num_batches
        
        if transformer_type == TransformerType.BAYESIAN:
            avg_train_bayesian_kl_loss = train_bayesian_kl_loss / num_batches
        
        # Validation
        model.eval()
        val_epoch_loss = 0.0
        val_rec_loss = 0.0
        val_kl_loss = 0.0
        
        if transformer_type == TransformerType.BAYESIAN:
            val_bayesian_kl_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            for batch in pbar:
                # Move batch to device
                visual_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in batch['visual_features'].items()}
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    visual_features=visual_features,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Update metrics
                val_epoch_loss += outputs['loss'].item()
                val_rec_loss += outputs['rec_loss'].item()
                
                if transformer_type == TransformerType.NORMAL:
                    val_kl_loss += outputs['kl_loss'].item()
                    pbar.set_postfix({
                        'loss': outputs['loss'].item(),
                        'rec_loss': outputs['rec_loss'].item(),
                        'kl_loss': outputs['kl_loss'].item()
                    })
                else:  # Bayesian
                    val_kl_loss += outputs['cvae_kl_loss'].item()
                    val_bayesian_kl_loss += outputs['bayesian_kl_loss'].item()
                    pbar.set_postfix({
                        'loss': outputs['loss'].item(),
                        'rec_loss': outputs['rec_loss'].item(),
                        'cvae_kl': outputs['cvae_kl_loss'].item(),
                        'bayes_kl': outputs['bayesian_kl_loss'].item()
                    })
        
        # Calculate average validation losses
        num_val_batches = len(val_loader)
        avg_val_loss = val_epoch_loss / num_val_batches
        avg_val_rec_loss = val_rec_loss / num_val_batches
        avg_val_kl_loss = val_kl_loss / num_val_batches
        
        if transformer_type == TransformerType.BAYESIAN:
            avg_val_bayesian_kl_loss = val_bayesian_kl_loss / num_val_batches
        
        # Save losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print epoch summary
        if transformer_type == TransformerType.NORMAL:
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Rec Loss: {avg_train_rec_loss:.4f}, KL Loss: {avg_train_kl_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Rec Loss: {avg_val_rec_loss:.4f}, KL Loss: {avg_val_kl_loss:.4f}")
        else:  # Bayesian
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Rec: {avg_train_rec_loss:.4f}, CVAE KL: {avg_train_kl_loss:.4f}, Bayes KL: {avg_train_bayesian_kl_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Rec: {avg_val_rec_loss:.4f}, CVAE KL: {avg_val_kl_loss:.4f}, Bayes KL: {avg_val_bayesian_kl_loss:.4f}")
        
        # Save checkpoint if it's the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.output_dir, f'{args.model_type}_unified_model_best.pt')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
                'model_type': args.model_type
            }, checkpoint_path)
            print(f"  Saved best model checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'{args.model_type}_unified_model_epoch{epoch+1}.pt')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
                'model_type': args.model_type
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {args.model_type.title()} Model')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = os.path.join(args.output_dir, f'{args.model_type}_unified_loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

def compare_models_demo(args):
    """
    Demo function to compare both model types on the same data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = VCGDataset(
        split='test',
        annotations_json=args.annotations_json,
        features_dir=args.features_dir,
        max_seq_len=args.max_seq_len
    )
    
    test_loader = VCGDataLoader(
        dataset=test_dataset,
        batch_size=1,  # Use batch size 1 for demo
        shuffle=False,
        num_workers=0,
        collate_fn=vcg_collate_fn
    )
    
    # Create model manager
    manager = ModelManager()
    vocab_size = len(test_dataset.tokenizer)
    
    # Create both models
    print("Creating models...")
    manager.create_and_add_model("normal", vocab_size, "normal")
    manager.create_and_add_model("bayesian", vocab_size, "bayesian")
    
    # Load checkpoints if provided
    if hasattr(args, 'normal_checkpoint') and args.normal_checkpoint:
        print(f"Loading normal model from {args.normal_checkpoint}")
        normal_checkpoint = torch.load(args.normal_checkpoint, map_location=device)
        manager.models["normal"].load_state_dict(normal_checkpoint['model_state_dict'])
    
    if hasattr(args, 'bayesian_checkpoint') and args.bayesian_checkpoint:
        print(f"Loading bayesian model from {args.bayesian_checkpoint}")
        bayesian_checkpoint = torch.load(args.bayesian_checkpoint, map_location=device)
        manager.models["bayesian"].load_state_dict(bayesian_checkpoint['model_state_dict'])
    
    # Move models to device
    for model in manager.models.values():
        model.to(device)
        model.eval()
    
    # Demo generation on a few samples
    print("\\nComparing model outputs...")
    num_samples = min(3, len(test_loader))
    
    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break
        
        print(f"\\n--- Sample {i+1} ---")
        
        # Move batch to device
        visual_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch['visual_features'].items()}
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Use first token as prompt
        prompt_ids = input_ids[:, 0:1]
        
        # Reference text
        reference_text = test_dataset.tokenizer.decode(labels[0].tolist(), skip_special_tokens=True)
        print(f"Reference: {reference_text}")
        
        # Compare models
        results = manager.compare_models(
            visual_features, 
            prompt_ids,
            max_length=args.max_seq_len,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=1
        )
        
        # Display results
        for model_name, result in results.items():
            generated_text = test_dataset.tokenizer.decode(
                result['generated_ids'][0].tolist(), 
                skip_special_tokens=True
            )
            print(f"{model_name.title()}: {generated_text}")
            
            if 'uncertainty' in result:
                avg_uncertainty = torch.mean(result['uncertainty']).item()
                print(f"  Average uncertainty: {avg_uncertainty:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Unified Training for Visual Backstory Generation')
    
    # Model selection
    parser.add_argument('--model_type', type=str, choices=['normal', 'bayesian'], 
                       default='normal', help='Type of transformer to use')
    parser.add_argument('--mode', type=str, choices=['train', 'compare'], 
                       default='train', help='Mode: train a model or compare models')
    
    # Data parameters
    parser.add_argument('--annotations_json', type=str, required=True,
                       help='Path to annotations JSON file')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing visual features')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension of the model')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Loss weights
    parser.add_argument('--kl_weight', type=float, default=0.01,
                       help='Weight for KL divergence loss')
    parser.add_argument('--bayesian_kl_weight', type=float, default=0.001,
                       help='Weight for Bayesian KL divergence loss (Bayesian only)')
    
    # Bayesian-specific parameters
    parser.add_argument('--prior_sigma_1', type=float, default=1.0,
                       help='First component of scale mixture prior')
    parser.add_argument('--prior_sigma_2', type=float, default=0.0025,
                       help='Second component of scale mixture prior')
    parser.add_argument('--prior_pi', type=float, default=0.5,
                       help='Mixture weight for scale mixture prior')
    
    # Comparison mode parameters
    parser.add_argument('--normal_checkpoint', type=str, default=None,
                       help='Path to normal model checkpoint (for comparison)')
    parser.add_argument('--bayesian_checkpoint', type=str, default=None,
                       help='Path to bayesian model checkpoint (for comparison)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        train_unified_model(args)
    elif args.mode == 'compare':
        compare_models_demo(args)

if __name__ == "__main__":
    main()
