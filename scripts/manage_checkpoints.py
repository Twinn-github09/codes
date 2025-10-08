"""
Helper script to list and manage model checkpoints
"""

import os
import argparse
import torch
from pathlib import Path
from datetime import datetime

def list_checkpoints(checkpoint_dir):
    """List all available checkpoints in a directory"""
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} does not exist!")
        return
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            checkpoint_files.append((file, file_size, modified_time, file_path))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"\\n=== AVAILABLE CHECKPOINTS IN {checkpoint_dir} ===")
    print(f"{'Filename':<40} {'Size (MB)':<10} {'Modified':<20} {'Info'}")
    print("-" * 90)
    
    for file, size, modified_time, file_path in sorted(checkpoint_files, key=lambda x: x[2], reverse=True):
        try:
            # Try to load checkpoint info
            checkpoint = torch.load(file_path, map_location='cpu')
            info = f"Epoch {checkpoint.get('epoch', '?')}"
            if 'best_val_loss' in checkpoint:
                info += f", Val Loss: {checkpoint['best_val_loss']:.4f}"
            elif 'val_loss' in checkpoint:
                info += f", Val Loss: {checkpoint['val_loss']:.4f}"
            if 'transformer_type' in checkpoint:
                info += f", Type: {checkpoint['transformer_type']}"
        except:
            info = "Could not read checkpoint info"
        
        print(f"{file:<40} {size:<10.1f} {modified_time.strftime('%Y-%m-%d %H:%M'):<20} {info}")
    
    print("-" * 90)
    print(f"Total: {len(checkpoint_files)} checkpoints")
    
    # Suggest which one to use for resuming
    latest_checkpoint = None
    best_checkpoint = None
    
    for file, _, _, file_path in checkpoint_files:
        if 'latest' in file.lower():
            latest_checkpoint = file
        elif 'best' in file.lower():
            best_checkpoint = file
    
    print(f"\\n=== RECOMMENDATIONS ===")
    if latest_checkpoint:
        print(f"To resume training: --resume_from '{os.path.join(checkpoint_dir, latest_checkpoint)}'")
    if best_checkpoint:
        print(f"For evaluation: --checkpoint '{os.path.join(checkpoint_dir, best_checkpoint)}'")
    
    return checkpoint_files

def checkpoint_info(checkpoint_path):
    """Show detailed info about a specific checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} does not exist!")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        
        print(f"\\n=== CHECKPOINT INFO: {os.path.basename(checkpoint_path)} ===")
        print(f"File size: {file_size:.1f} MB")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Model type: {checkpoint.get('model_type', 'Unknown')}")
        print(f"Transformer type: {checkpoint.get('transformer_type', 'Unknown')}")
        
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\\nModel config:")
            print(f"  Hidden dim: {config.get('hidden_dim', 'Unknown')}")
            print(f"  Vocab size: {config.get('vocab_size', 'Unknown')}")
            print(f"  Max sequence length: {config.get('max_position_embeddings', 'Unknown')}")
        
        if 'history' in checkpoint and checkpoint['history']:
            history = checkpoint['history']
            print(f"\\nTraining history:")
            if 'train_loss' in history and history['train_loss']:
                print(f"  Train losses: {len(history['train_loss'])} epochs")
                print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            if 'val_loss' in history and history['val_loss']:
                print(f"  Val losses: {len(history['val_loss'])} epochs")
                print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        
        # Check if this checkpoint can be used for resuming
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']
        can_resume = all(key in checkpoint for key in required_keys)
        print(f"\\nCan resume training: {'Yes' if can_resume else 'No'}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage model checkpoints")
    parser.add_argument("--dir", default="models/filtered", help="Checkpoint directory")
    parser.add_argument("--info", help="Show detailed info about specific checkpoint")
    parser.add_argument("--list", action="store_true", default=True, help="List all checkpoints")
    
    args = parser.parse_args()
    
    if args.info:
        checkpoint_info(args.info)
    elif args.list:
        list_checkpoints(args.dir)

if __name__ == "__main__":
    main()