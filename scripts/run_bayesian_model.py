"""
Train and evaluate the Bayesian Visual Backstory Generation model.
This script provides functions for training, evaluation, and comparison
between the standard and Bayesian models.
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

# Import models
from models.backstory_model import VisualBackstoryGenerationModel, create_backstory_model_config
from models.bayesian_backstory_model import BayesianVisualBackstoryGenerationModel, create_bayesian_backstory_model_config
from dataloaders.vcg import VCGDataset, VCGDataLoader, vcg_collate_fn

def train_bayesian_model(args):
    """
    Train the Bayesian Visual Backstory Generation model
    
    Args:
        args: Command-line arguments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = VCGDataset(
        split='train',
        annotations_json=args.annotations_json,
        features_dir=args.features_dir,
        max_seq_len=args.max_seq_len
    )
    val_dataset = VCGDataset(
        split='val',
        annotations_json=args.annotations_json,
        features_dir=args.features_dir,
        max_seq_len=args.max_seq_len
    )
    
    # Create data loaders
    train_loader = VCGDataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=vcg_collate_fn
    )
    val_loader = VCGDataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vcg_collate_fn
    )
    
    # Create model
    vocab_size = len(train_dataset.tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    if args.model_type == 'standard':
        config = create_backstory_model_config(vocab_size)
        model = VisualBackstoryGenerationModel(config)
    else:  # bayesian
        config = create_bayesian_backstory_model_config(vocab_size)
        model = BayesianVisualBackstoryGenerationModel(config)
        
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_epoch_loss = 0.0
        train_rec_loss = 0.0
        train_kl_loss = 0.0
        
        if args.model_type == 'bayesian':
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
            
            # Calculate loss
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_epoch_loss += loss.item()
            train_rec_loss += outputs['rec_loss'].item()
            
            if args.model_type == 'standard':
                train_kl_loss += outputs['kl_loss'].item()
                pbar.set_postfix({
                    'loss': loss.item(),
                    'rec_loss': outputs['rec_loss'].item(),
                    'kl_loss': outputs['kl_loss'].item()
                })
            else:  # bayesian
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
        
        if args.model_type == 'bayesian':
            avg_train_bayesian_kl_loss = train_bayesian_kl_loss / num_batches
        
        # Validation
        model.eval()
        val_epoch_loss = 0.0
        val_rec_loss = 0.0
        val_kl_loss = 0.0
        
        if args.model_type == 'bayesian':
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
                
                if args.model_type == 'standard':
                    val_kl_loss += outputs['kl_loss'].item()
                    pbar.set_postfix({
                        'loss': outputs['loss'].item(),
                        'rec_loss': outputs['rec_loss'].item(),
                        'kl_loss': outputs['kl_loss'].item()
                    })
                else:  # bayesian
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
        
        if args.model_type == 'bayesian':
            avg_val_bayesian_kl_loss = val_bayesian_kl_loss / num_val_batches
        
        # Save losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print epoch summary
        if args.model_type == 'standard':
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Rec Loss: {avg_train_rec_loss:.4f}, KL Loss: {avg_train_kl_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Rec Loss: {avg_val_rec_loss:.4f}, KL Loss: {avg_val_kl_loss:.4f}")
        else:  # bayesian
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Rec: {avg_train_rec_loss:.4f}, CVAE KL: {avg_train_kl_loss:.4f}, Bayes KL: {avg_train_bayesian_kl_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Rec: {avg_val_rec_loss:.4f}, CVAE KL: {avg_val_kl_loss:.4f}, Bayes KL: {avg_val_bayesian_kl_loss:.4f}")
        
        # Save checkpoint if it's the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_type_str = 'standard' if args.model_type == 'standard' else 'bayesian'
            checkpoint_path = os.path.join(args.output_dir, f'{model_type_str}_backstory_model_best.pt')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Saved best model checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            model_type_str = 'standard' if args.model_type == 'standard' else 'bayesian'
            checkpoint_path = os.path.join(args.output_dir, f'{model_type_str}_backstory_model_epoch{epoch+1}.pt')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    model_type_str = 'standard' if args.model_type == 'standard' else 'bayesian'
    loss_plot_path = os.path.join(args.output_dir, f'{model_type_str}_loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")


def evaluate_bayesian_model(args):
    """
    Evaluate the Bayesian Visual Backstory Generation model
    
    Args:
        args: Command-line arguments
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
    
    # Create data loader
    test_loader = VCGDataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vcg_collate_fn
    )
    
    # Load checkpoint
    vocab_size = len(test_dataset.tokenizer)
    
    if args.model_type == 'standard':
        config = create_backstory_model_config(vocab_size)
        model = VisualBackstoryGenerationModel(config)
    else:  # bayesian
        config = create_bayesian_backstory_model_config(vocab_size)
        model = BayesianVisualBackstoryGenerationModel(config)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluation metrics
    test_loss = 0.0
    test_rec_loss = 0.0
    test_kl_loss = 0.0
    
    if args.model_type == 'bayesian':
        test_bayesian_kl_loss = 0.0
    
    # Generation results
    generated_examples = []
    
    with torch.no_grad():
        # Calculate test loss
        pbar = tqdm(test_loader, desc="Evaluating")
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
            test_loss += outputs['loss'].item()
            test_rec_loss += outputs['rec_loss'].item()
            
            if args.model_type == 'standard':
                test_kl_loss += outputs['kl_loss'].item()
            else:  # bayesian
                test_kl_loss += outputs['cvae_kl_loss'].item()
                test_bayesian_kl_loss += outputs['bayesian_kl_loss'].item()
            
            # Generate text for sample images
            if len(generated_examples) < args.num_examples:
                # Use first token as prompt (BOS token)
                prompt_ids = input_ids[:, 0:1]
                
                if args.model_type == 'standard':
                    generated_ids = model.generate(
                        visual_features=visual_features,
                        input_ids=prompt_ids,
                        max_length=args.max_seq_len,
                        do_sample=True,
                        top_p=0.9,
                        temperature=args.temperature,
                        num_return_sequences=args.num_generations_per_example
                    )
                else:  # bayesian
                    # Generate with uncertainty estimation
                    generated_ids, uncertainty = model.generate(
                        visual_features=visual_features,
                        input_ids=prompt_ids,
                        max_length=args.max_seq_len,
                        do_sample=True,
                        top_p=0.9,
                        temperature=args.temperature,
                        num_return_sequences=args.num_generations_per_example,
                        output_uncertainty=True,
                        num_uncertainty_samples=5
                    )
                
                # Convert IDs to text
                for i in range(min(args.batch_size, len(batch['visual_features']['image_ids']))):
                    image_id = batch['visual_features']['image_ids'][i]
                    reference_text = test_dataset.tokenizer.decode(labels[i].tolist(), skip_special_tokens=True)
                    
                    generated_texts = []
                    for j in range(args.num_generations_per_example):
                        idx = i * args.num_generations_per_example + j
                        if idx < len(generated_ids):
                            gen_text = test_dataset.tokenizer.decode(generated_ids[idx].tolist(), skip_special_tokens=True)
                            generated_texts.append(gen_text)
                    
                    example = {
                        'image_id': image_id,
                        'reference': reference_text,
                        'generated': generated_texts
                    }
                    
                    if args.model_type == 'bayesian':
                        # Add uncertainty information for each token in the first generation
                        if len(generated_texts) > 0:
                            # Get the decoded tokens
                            tokens = test_dataset.tokenizer.convert_ids_to_tokens(generated_ids[i * args.num_generations_per_example].tolist())
                            
                            # Get uncertainty for each token (from the first generation of this example)
                            token_uncertainties = uncertainty[:, i * args.num_generations_per_example].tolist()
                            
                            # Trim to actual sequence length
                            tokens = tokens[:len(token_uncertainties)]
                            
                            # Create a list of [token, uncertainty] pairs
                            uncertainty_info = list(zip(tokens, token_uncertainties))
                            example['token_uncertainties'] = uncertainty_info
                    
                    generated_examples.append(example)
                    
                    # Break if we have enough examples
                    if len(generated_examples) >= args.num_examples:
                        break
    
    # Calculate average test metrics
    num_batches = len(test_loader)
    avg_test_loss = test_loss / num_batches
    avg_test_rec_loss = test_rec_loss / num_batches
    avg_test_kl_loss = test_kl_loss / num_batches
    
    if args.model_type == 'bayesian':
        avg_test_bayesian_kl_loss = test_bayesian_kl_loss / num_batches
        print(f"Test Loss: {avg_test_loss:.4f}, Rec: {avg_test_rec_loss:.4f}, CVAE KL: {avg_test_kl_loss:.4f}, Bayes KL: {avg_test_bayesian_kl_loss:.4f}")
    else:
        print(f"Test Loss: {avg_test_loss:.4f}, Rec: {avg_test_rec_loss:.4f}, KL: {avg_test_kl_loss:.4f}")
    
    # Save generated examples
    model_type_str = 'standard' if args.model_type == 'standard' else 'bayesian'
    output_file = os.path.join(args.output_dir, f'{model_type_str}_generated_examples.json')
    with open(output_file, 'w') as f:
        json.dump(generated_examples, f, indent=2)
    
    print(f"Generated examples saved to {output_file}")


def compare_models(args):
    """
    Compare standard and Bayesian models' generations
    
    Args:
        args: Command-line arguments
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
    
    # Create data loader
    test_loader = VCGDataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vcg_collate_fn
    )
    
    # Load both models
    vocab_size = len(test_dataset.tokenizer)
    
    # Standard model
    std_config = create_backstory_model_config(vocab_size)
    std_model = VisualBackstoryGenerationModel(std_config)
    std_checkpoint = torch.load(args.std_checkpoint_path, map_location=device)
    std_model.load_state_dict(std_checkpoint['model_state_dict'])
    std_model.to(device)
    std_model.eval()
    
    # Bayesian model
    bayes_config = create_bayesian_backstory_model_config(vocab_size)
    bayes_model = BayesianVisualBackstoryGenerationModel(bayes_config)
    bayes_checkpoint = torch.load(args.bayes_checkpoint_path, map_location=device)
    bayes_model.load_state_dict(bayes_checkpoint['model_state_dict'])
    bayes_model.to(device)
    bayes_model.eval()
    
    # Comparison results
    comparison_examples = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Comparing models")
        for batch in pbar:
            if len(comparison_examples) >= args.num_examples:
                break
                
            # Move batch to device
            visual_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch['visual_features'].items()}
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Use first token as prompt (BOS token)
            prompt_ids = input_ids[:, 0:1]
            
            # Generate with standard model
            std_generated_ids = std_model.generate(
                visual_features=visual_features,
                input_ids=prompt_ids,
                max_length=args.max_seq_len,
                do_sample=True,
                top_p=0.9,
                temperature=args.temperature,
                num_return_sequences=args.num_generations_per_example
            )
            
            # Generate with Bayesian model
            bayes_generated_ids, uncertainty = bayes_model.generate(
                visual_features=visual_features,
                input_ids=prompt_ids,
                max_length=args.max_seq_len,
                do_sample=True,
                top_p=0.9,
                temperature=args.temperature,
                num_return_sequences=args.num_generations_per_example,
                output_uncertainty=True,
                num_uncertainty_samples=5
            )
            
            # Convert IDs to text and build comparison
            for i in range(min(args.batch_size, len(batch['visual_features']['image_ids']))):
                if len(comparison_examples) >= args.num_examples:
                    break
                    
                image_id = batch['visual_features']['image_ids'][i]
                reference_text = test_dataset.tokenizer.decode(labels[i].tolist(), skip_special_tokens=True)
                
                # Standard model generations
                std_generated_texts = []
                for j in range(args.num_generations_per_example):
                    idx = i * args.num_generations_per_example + j
                    if idx < len(std_generated_ids):
                        gen_text = test_dataset.tokenizer.decode(std_generated_ids[idx].tolist(), skip_special_tokens=True)
                        std_generated_texts.append(gen_text)
                
                # Bayesian model generations
                bayes_generated_texts = []
                for j in range(args.num_generations_per_example):
                    idx = i * args.num_generations_per_example + j
                    if idx < len(bayes_generated_ids):
                        gen_text = test_dataset.tokenizer.decode(bayes_generated_ids[idx].tolist(), skip_special_tokens=True)
                        bayes_generated_texts.append(gen_text)
                
                # Get uncertainty information for the first Bayesian generation
                tokens = test_dataset.tokenizer.convert_ids_to_tokens(bayes_generated_ids[i * args.num_generations_per_example].tolist())
                token_uncertainties = uncertainty[:, i * args.num_generations_per_example].tolist()
                tokens = tokens[:len(token_uncertainties)]
                uncertainty_info = list(zip(tokens, token_uncertainties))
                
                example = {
                    'image_id': image_id,
                    'reference': reference_text,
                    'standard_generated': std_generated_texts,
                    'bayesian_generated': bayes_generated_texts,
                    'token_uncertainties': uncertainty_info
                }
                
                comparison_examples.append(example)
    
    # Save comparison results
    output_file = os.path.join(args.output_dir, 'model_comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump(comparison_examples, f, indent=2)
    
    print(f"Comparison results saved to {output_file}")
    
    # Create uncertainty visualization
    if comparison_examples:
        plt.figure(figsize=(15, 10))
        
        for i, example in enumerate(comparison_examples[:5]):  # Visualize first 5 examples
            plt.subplot(5, 1, i+1)
            
            tokens = [t[0] for t in example['token_uncertainties'] if not t[0].startswith('##')]
            uncertainties = [t[1] for t in example['token_uncertainties'] if not t[0].startswith('##')]
            
            if len(tokens) > 30:
                tokens = tokens[:30]
                uncertainties = uncertainties[:30]
            
            plt.bar(range(len(tokens)), uncertainties)
            plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
            plt.ylabel('Uncertainty')
            plt.title(f"Example {i+1} - Token Uncertainty")
            plt.tight_layout()
        
        uncertainty_plot_path = os.path.join(args.output_dir, 'token_uncertainty_visualization.png')
        plt.savefig(uncertainty_plot_path)
        print(f"Uncertainty visualization saved to {uncertainty_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate Bayesian Visual Backstory Generation model")
    
    # Common arguments
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'compare'],
                        help='Mode: train, evaluate, or compare models')
    parser.add_argument('--model_type', type=str, default='bayesian', choices=['standard', 'bayesian'],
                        help='Model type: standard or bayesian')
    parser.add_argument('--annotations_json', type=str, required=True,
                        help='Path to annotations JSON file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Path to extracted features directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and results')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps for learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    
    # Evaluation/comparison arguments
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to model checkpoint for evaluation')
    parser.add_argument('--std_checkpoint_path', type=str,
                        help='Path to standard model checkpoint for comparison')
    parser.add_argument('--bayes_checkpoint_path', type=str,
                        help='Path to Bayesian model checkpoint for comparison')
    parser.add_argument('--num_examples', type=int, default=20,
                        help='Number of examples for evaluation/comparison')
    parser.add_argument('--num_generations_per_example', type=int, default=3,
                        help='Number of sequences to generate per example')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        train_bayesian_model(args)
    elif args.mode == 'evaluate':
        if args.checkpoint_path is None:
            raise ValueError("Checkpoint path is required for evaluation mode")
        evaluate_bayesian_model(args)
    elif args.mode == 'compare':
        if args.std_checkpoint_path is None or args.bayes_checkpoint_path is None:
            raise ValueError("Both standard and Bayesian checkpoint paths are required for comparison mode")
        compare_models(args)
