"""
Modified Training Script for Filtered Backstory Dataset.
This script trains models using only annotations that have corresponding feature files.
"""

import sys
import os

# Add parent directory to path so we can import from models and dataloaders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup, GPT2Tokenizer

# Import unified model system
from models.unified_model import (
    UnifiedVisualBackstoryModel, TransformerType, 
    create_unified_config, create_model, ModelManager
)
from dataloaders.filtered_dataset import create_filtered_dataloaders

def train_filtered_model(args):
    """
    Train model using filtered dataset that only includes annotations with features
    
    Args:
        args: Command-line arguments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Detailed GPU logging
    if torch.cuda.is_available():
        print(f"GPU Details:")
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - PyTorch Version: {torch.__version__}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  - Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        print(f"  - GPU Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    else:
        print("WARNING: CUDA not available, using CPU")
    
    print(f"Training {args.model_type} transformer model")
    print(f"Features directory: {args.features_dir}")
    
    # Initialize tokenizer first
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create filtered datasets and dataloaders
    train_loader, val_loader = create_filtered_dataloaders(
        annotations_json=args.annotations_json,
        val_annotations_json=args.val_annotations_json,
        features_dir=args.features_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=getattr(args, 'num_workers', 4)
    )
    
    # Create model
    vocab_size = tokenizer.vocab_size
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
            'prior_sigma_1': getattr(args, 'prior_sigma_1', 1.0),
            'prior_sigma_2': getattr(args, 'prior_sigma_2', 0.1),
            'prior_pi': getattr(args, 'prior_pi', 0.25)
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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          weight_decay=getattr(args, 'weight_decay', 0.01))
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=getattr(args, 'warmup_steps', 0),
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rec_loss': [],
        'val_rec_loss': []
    }
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n=== RESUMING FROM CHECKPOINT ===")
        print(f"Loading checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'history' in checkpoint:
            history = checkpoint['history']
        
        print(f"Resumed from epoch {checkpoint['epoch'] + 1}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")
        print("================================\n")
    elif args.resume_from:
        print(f"Warning: Checkpoint file {args.resume_from} not found. Starting fresh training.")
        print("========================================\n")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_epoch_loss = 0.0
        train_rec_loss = 0.0
        train_kl_loss = 0.0
        
        if transformer_type == TransformerType.BAYESIAN:
            train_bayesian_kl_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Unpack batch tuple: (event_inference_example, labels, features, boxes, boxes_mask, objects, segments, person_ids, subject_ids)
            input_ids, labels, img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = batch
            
            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            img_feats = img_feats.to(device)
            boxes = boxes.to(device)
            boxes_mask = boxes_mask.to(device)
            objects = objects.to(device)
            segments = segments.to(device)
            person_ids = person_ids.to(device)
            subject_ids = subject_ids.to(device)
            
            # Log GPU usage for first batch to verify GPU is being used
            if batch_idx == 0 and torch.cuda.is_available():
                print(f"\n=== GPU VERIFICATION (Batch 0) ===")
                print(f"Input tensor device: {input_ids.device}")
                print(f"Visual features device: {img_feats.device}")
                print(f"GPU Memory after moving tensors: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
                print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e6:.1f} MB")
                print("===================================\n")
            
            # Forward pass - Prepare visual features dict for the detector
            visual_features_dict = {
                'image_features': img_feats,  # Raw features (what backstory_model expects)
                'boxes': boxes,               # Dummy boxes from dataset
                'box_mask': boxes_mask,       # Mask for valid boxes (what detector expects)
                'class_ids': objects          # Object class IDs (what backstory_model expects)
            }
            
            outputs = model(
                visual_features=visual_features_dict,
                input_ids=input_ids,
                attention_mask=None,  # Will be computed inside model if needed
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Log GPU usage after forward pass for first batch
            if batch_idx == 0 and torch.cuda.is_available():
                print(f"GPU Memory after forward pass: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
                print(f"Loss tensor device: {loss.device}")
                print(f"Model parameters on GPU: {next(model.parameters()).device}")
                print("===================================")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(args, 'gradient_clip_norm') and args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            
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
                # Unpack batch tuple: (event_inference_example, labels, features, boxes, boxes_mask, objects, segments, person_ids, subject_ids)
                input_ids, labels, img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = batch
                
                # Move to device
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                img_feats = img_feats.to(device)
                boxes = boxes.to(device)
                boxes_mask = boxes_mask.to(device)
                objects = objects.to(device)
                segments = segments.to(device)
                person_ids = person_ids.to(device)
                subject_ids = subject_ids.to(device)
                
                # Forward pass - Prepare visual features dict
                visual_features_dict = {
                    'image_features': img_feats,
                    'boxes': boxes,
                    'box_mask': boxes_mask,
                    'class_ids': objects
                }
                
                outputs = model(
                    visual_features=visual_features_dict,
                    input_ids=input_ids,
                    attention_mask=None,
                    labels=labels
                )
                
                # Update metrics
                val_epoch_loss += outputs['loss'].item()
                val_rec_loss += outputs['rec_loss'].item()
                
                if transformer_type == TransformerType.NORMAL:
                    val_kl_loss += outputs['kl_loss'].item()
                else:  # Bayesian
                    val_kl_loss += outputs['cvae_kl_loss'].item()
                    val_bayesian_kl_loss += outputs['bayesian_kl_loss'].item()
        
        # Calculate validation averages
        avg_val_loss = val_epoch_loss / len(val_loader)
        avg_val_rec_loss = val_rec_loss / len(val_loader)
        avg_val_kl_loss = val_kl_loss / len(val_loader)
        
        if transformer_type == TransformerType.BAYESIAN:
            avg_val_bayesian_kl_loss = val_bayesian_kl_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rec_loss'].append(avg_train_rec_loss)
        history['val_rec_loss'].append(avg_val_rec_loss)
        
        # Print epoch summary
        print(f"\\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Train Rec Loss: {avg_train_rec_loss:.4f}")
        print(f"  Val Rec Loss: {avg_val_rec_loss:.4f}")
        
        if transformer_type == TransformerType.BAYESIAN:
            print(f"  Train Bayesian KL: {avg_train_bayesian_kl_loss:.4f}")
            print(f"  Val Bayesian KL: {avg_val_bayesian_kl_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(args.output_dir, f'{args.model_type}_backstory_model_best.pt')
            
            # Prepare comprehensive loss components for saving
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'reconstruction_loss': avg_val_rec_loss,
                'kl_loss': avg_val_kl_loss,
                'train_reconstruction_loss': avg_train_rec_loss,
                'train_kl_loss': avg_train_kl_loss,
                'learning_rate': scheduler.get_last_lr()[0] if scheduler else args.learning_rate,
                'kl_weight': getattr(args, 'kl_weight', 1.0),
                'config': config.__dict__,
                'model_type': args.model_type,
                'transformer_type': transformer_type.value,
                'history': history,
                'training_info': {
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'kl_weight': getattr(args, 'kl_weight', 1.0),
                    'num_epochs': args.num_epochs,
                    'warmup_steps': getattr(args, 'warmup_steps', 0),
                    'reconstruction_loss': avg_val_rec_loss,
                    'kl_divergence': avg_val_kl_loss,
                    'train_reconstruction_loss': avg_train_rec_loss,
                    'train_kl_divergence': avg_train_kl_loss
                }
            }
            
            # Add Bayesian-specific loss components if applicable
            if transformer_type == TransformerType.BAYESIAN:
                checkpoint_data['bayesian_kl_loss'] = avg_val_bayesian_kl_loss
                checkpoint_data['train_bayesian_kl_loss'] = avg_train_bayesian_kl_loss
                checkpoint_data['training_info']['bayesian_kl_loss'] = avg_val_bayesian_kl_loss
                checkpoint_data['training_info']['train_bayesian_kl_loss'] = avg_train_bayesian_kl_loss
            
            torch.save(checkpoint_data, model_path)
            print(f"  Saved best model to {model_path} (with loss components)")
        
        # Save regular checkpoint
        if hasattr(args, 'save_every') and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'{args.model_type}_backstory_model_epoch{epoch+1}.pt')
            
            # Prepare comprehensive loss components for saving
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'reconstruction_loss': avg_val_rec_loss,
                'kl_loss': avg_val_kl_loss,
                'train_reconstruction_loss': avg_train_rec_loss,
                'train_kl_loss': avg_train_kl_loss,
                'learning_rate': scheduler.get_last_lr()[0] if scheduler else args.learning_rate,
                'kl_weight': getattr(args, 'kl_weight', 1.0),
                'config': config.__dict__,
                'model_type': args.model_type,
                'transformer_type': transformer_type.value,
                'history': history,
                'training_info': {
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'kl_weight': getattr(args, 'kl_weight', 1.0),
                    'num_epochs': args.num_epochs,
                    'warmup_steps': getattr(args, 'warmup_steps', 0),
                    'reconstruction_loss': avg_val_rec_loss,
                    'kl_divergence': avg_val_kl_loss,
                    'train_reconstruction_loss': avg_train_rec_loss,
                    'train_kl_divergence': avg_train_kl_loss
                }
            }
            
            # Add Bayesian-specific loss components if applicable
            if transformer_type == TransformerType.BAYESIAN:
                checkpoint_data['bayesian_kl_loss'] = avg_val_bayesian_kl_loss
                checkpoint_data['train_bayesian_kl_loss'] = avg_train_bayesian_kl_loss
                checkpoint_data['training_info']['bayesian_kl_loss'] = avg_val_bayesian_kl_loss
                checkpoint_data['training_info']['train_bayesian_kl_loss'] = avg_train_bayesian_kl_loss
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path} (with loss components)")
        
        # Always save latest checkpoint (for resuming)
        latest_path = os.path.join(args.output_dir, f'{args.model_type}_backstory_model_latest.pt')
        
        # Prepare comprehensive loss components for saving
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'reconstruction_loss': avg_val_rec_loss,
            'kl_loss': avg_val_kl_loss,
            'train_reconstruction_loss': avg_train_rec_loss,
            'train_kl_loss': avg_train_kl_loss,
            'learning_rate': scheduler.get_last_lr()[0] if scheduler else args.learning_rate,
            'kl_weight': getattr(args, 'kl_weight', 1.0),
            'config': config.__dict__,
            'model_type': args.model_type,
            'transformer_type': transformer_type.value,
            'history': history,
            'training_info': {
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'kl_weight': getattr(args, 'kl_weight', 1.0),
                'num_epochs': args.num_epochs,
                'warmup_steps': getattr(args, 'warmup_steps', 0),
                'reconstruction_loss': avg_val_rec_loss,
                'kl_divergence': avg_val_kl_loss,
                'train_reconstruction_loss': avg_train_rec_loss,
                'train_kl_divergence': avg_train_kl_loss
            }
        }
        
        # Add Bayesian-specific loss components if applicable
        if transformer_type == TransformerType.BAYESIAN:
            checkpoint_data['bayesian_kl_loss'] = avg_val_bayesian_kl_loss
            checkpoint_data['train_bayesian_kl_loss'] = avg_train_bayesian_kl_loss
            checkpoint_data['training_info']['bayesian_kl_loss'] = avg_val_bayesian_kl_loss
            checkpoint_data['training_info']['train_bayesian_kl_loss'] = avg_train_bayesian_kl_loss
        
        torch.save(checkpoint_data, latest_path)
    
    # Save final model
    final_path = os.path.join(args.output_dir, f'{args.model_type}_backstory_model_final.pt')
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': avg_val_loss,
        'best_val_loss': best_val_loss,
        'config': config.__dict__,
        'model_type': args.model_type,
        'transformer_type': transformer_type.value,
        'history': history
    }, final_path)
    
    print(f"\\n=== TRAINING COMPLETE ===")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, f'{args.model_type}_backstory_model_best.pt')}")
    print(f"Latest checkpoint: {os.path.join(args.output_dir, f'{args.model_type}_backstory_model_latest.pt')}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("==========================")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return history

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Visual Backstory Generation Model with Filtered Dataset")
    
    # Dataset arguments
    parser.add_argument("--annotations_json", required=True,
                        help="Path to training annotations JSON file")
    parser.add_argument("--val_annotations_json", required=True,
                        help="Path to validation annotations JSON file")
    parser.add_argument("--features_dir", required=True,
                        help="Directory containing feature .pkl files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for checkpoints")
    
    # Model arguments
    parser.add_argument("--model_type", choices=['normal', 'bayesian'], default='normal',
                        help="Type of transformer to use")
    parser.add_argument("--hidden_dim", type=int, default=768,
                        help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Warmup steps")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    
    # Loss weights
    parser.add_argument("--kl_weight", type=float, default=0.01,
                        help="KL divergence loss weight")
    parser.add_argument("--bayesian_kl_weight", type=float, default=0.001,
                        help="Bayesian KL divergence loss weight")
    
    # Other arguments
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--save_every", type=int, default=2,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Train model
    history = train_filtered_model(args)

if __name__ == "__main__":
    main()