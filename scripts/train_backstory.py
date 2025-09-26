"""
Training script for Visual Backstory Generation model.
"""

import os
import json
import pickle
import argparse
import logging
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup
from dataloaders.tokenizers import VisualCometTokenizer
from models.backstory_model import VisualBackstoryGenerationModel, create_backstory_model_config

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class BackstoryDataset(Dataset):
    """
    Dataset for Visual Backstory Generation task.
    Loads processed features and annotations.
    """
    
    def __init__(self, data_file, features_dir, tokenizer, max_seq_len=128):
        """
        Initialize the dataset
        
        Args:
            data_file: Path to the JSON data file
            features_dir: Base directory for features
            tokenizer: Tokenizer for text processing
            max_seq_len: Maximum sequence length
        """
        self.data = json.load(open(data_file))
        self.features_dir = features_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        logger.info(f"Loaded {len(self.data)} examples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Load features
        feature_path = os.path.join(self.features_dir, example['feature_path'])
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
        
        # Process text inputs
        place = example['place']
        event = example['event']
        backstory = example['backstory']
        
        # Format text input as: "Place: {place} Event: {event} Before: "
        # The model will generate the backstory
        text_input = f"Place: {place} Event: {event} Before: "
        
        # Tokenize input text
        input_ids = self.tokenizer.encode(text_input)
        
        # Tokenize backstory (target)
        backstory_ids = self.tokenizer.encode(backstory)
        
        # Prepare inputs for CVAE training
        # For training, we include the backstory in the input but mask it for prediction
        combined_ids = input_ids + backstory_ids + [self.tokenizer.eos_token_id]
        
        # Truncate if too long
        if len(combined_ids) > self.max_seq_len:
            combined_ids = combined_ids[:self.max_seq_len]
        
        # Create labels (shift by 1)
        labels = [-100] * len(input_ids) + backstory_ids + [self.tokenizer.eos_token_id]
        
        # Truncate if too long
        if len(labels) > self.max_seq_len:
            labels = labels[:self.max_seq_len]
        
        # Pad if needed
        attention_mask = [1] * len(combined_ids)
        padding_length = self.max_seq_len - len(combined_ids)
        
        if padding_length > 0:
            combined_ids = combined_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
        
        # Convert to tensors
        input_tensor = torch.tensor(combined_ids, dtype=torch.long)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Get visual features
        image_features = torch.tensor(features['image_features'], dtype=torch.float)
        object_features = torch.tensor(features['object_features'], dtype=torch.float)
        boxes = torch.tensor(features['boxes'], dtype=torch.float)
        
        # Create box mask (1 for valid boxes)
        box_mask = torch.ones(object_features.shape[0] + 1, dtype=torch.long)
        
        # Class IDs for conditioning
        if 'class_ids' in features:
            class_ids = torch.tensor(features['class_ids'], dtype=torch.long)
        else:
            # Default to zeros if not available
            class_ids = torch.zeros(object_features.shape[0] + 1, dtype=torch.long)
        
        # Combine all visual features
        visual_inputs = {
            'image_features': image_features,
            'object_features': object_features,
            'boxes': boxes,
            'box_mask': box_mask,
            'class_ids': class_ids
        }
        
        return {
            'input_ids': input_tensor,
            'attention_mask': mask_tensor,
            'labels': labels_tensor,
            'visual_inputs': visual_inputs,
        }


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    """Evaluate the model on the validation set"""
    model.eval()
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            visual_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch['visual_inputs'].items()}
            
            # Forward pass
            outputs = model(
                visual_features=visual_inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Accumulate losses
            loss = outputs['loss']
            total_loss += loss.item() * input_ids.size(0)
            total_rec_loss += outputs['rec_loss'].item() * input_ids.size(0)
            total_kl_loss += outputs['kl_loss'].item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    # Calculate average losses
    avg_loss = total_loss / total_samples
    avg_rec_loss = total_rec_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples
    
    return {
        'loss': avg_loss,
        'rec_loss': avg_rec_loss,
        'kl_loss': avg_kl_loss
    }


def generate_samples(model, dataloader, tokenizer, device, num_samples=5):
    """Generate and print sample backstories"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            visual_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch['visual_inputs'].items()}
            
            # Get text input (without backstory)
            input_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            prefix_pos = input_text.find("Before: ")
            if prefix_pos != -1:
                prefix = input_text[:prefix_pos + len("Before: ")]
                
                # Tokenize just the prefix
                prefix_ids = tokenizer.encode(prefix)
                prefix_ids_tensor = torch.tensor([prefix_ids], device=device)
                
                # Generate text
                generated_ids = model.generate(
                    visual_features=visual_inputs,
                    input_ids=prefix_ids_tensor,
                    max_length=50,
                    temperature=0.9,
                    top_p=0.9,
                    do_sample=True
                )
                
                # Decode generated text
                generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
                
                # Get ground truth
                ground_truth = input_text[prefix_pos + len("Before: "):]
                
                samples.append({
                    'prefix': prefix,
                    'generated': generated_text[len(prefix):],
                    'ground_truth': ground_truth
                })
    
    return samples


def train(args):
    """Main training function"""
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = VisualCometTokenizer.from_pretrained('gpt2')
    
    # Add special tokens for backstory generation
    special_tokens = {
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<|place|>', '<|event|>', '<|before|>']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Create datasets
    train_dataset = BackstoryDataset(
        os.path.join(args.data_dir, 'processed', 'train_dataset.json'),
        args.data_dir,
        tokenizer,
        max_seq_len=args.max_seq_len
    )
    
    val_dataset = BackstoryDataset(
        os.path.join(args.data_dir, 'processed', 'val_dataset.json'),
        args.data_dir,
        tokenizer,
        max_seq_len=args.max_seq_len
    )
    
    # Create data loaders
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=None
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=None
    )
    
    # Initialize model
    config = create_backstory_model_config(tokenizer.vocab_size)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.kl_weight = args.kl_weight
    
    model = VisualBackstoryGenerationModel(config)
    model.to(device)
    
    # Initialize optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Calculate training steps
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_rec_loss = 0
        epoch_kl_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            visual_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch['visual_inputs'].items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                visual_features=visual_inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get losses
            loss = outputs['loss']
            rec_loss = outputs['rec_loss']
            kl_loss = outputs['kl_loss']
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Log metrics
            epoch_loss += loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            global_step += 1
            
            # Log to TensorBoard
            if global_step % args.logging_steps == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/rec_loss', rec_loss.item(), global_step)
                writer.add_scalar('train/kl_loss', kl_loss.item(), global_step)
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
        
        # Calculate average epoch losses
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_rec_loss = epoch_rec_loss / len(train_dataloader)
        avg_epoch_kl_loss = epoch_kl_loss / len(train_dataloader)
        
        logger.info(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Rec Loss: {avg_epoch_rec_loss:.4f}, KL Loss: {avg_epoch_kl_loss:.4f}")
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, device)
        logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, Rec Loss: {val_metrics['rec_loss']:.4f}, KL Loss: {val_metrics['kl_loss']:.4f}")
        
        # Log validation metrics
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/rec_loss', val_metrics['rec_loss'], epoch)
        writer.add_scalar('val/kl_loss', val_metrics['kl_loss'], epoch)
        
        # Generate samples
        if (epoch + 1) % args.generation_epochs == 0:
            samples = generate_samples(model, val_dataloader, tokenizer, device)
            
            # Log samples
            logger.info("\nGenerated Samples:")
            for i, sample in enumerate(samples):
                logger.info(f"Sample {i+1}:")
                logger.info(f"Prefix: {sample['prefix']}")
                logger.info(f"Generated: {sample['generated']}")
                logger.info(f"Ground Truth: {sample['ground_truth']}\n")
        
        # Save model checkpoint if it's the best so far
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            # Save model
            model_path = os.path.join(args.output_dir, 'best_model')
            os.makedirs(model_path, exist_ok=True)
            
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            logger.info(f"New best model saved to {model_path}")
        
        # Always save latest checkpoint
        model_path = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch+1}')
        os.makedirs(model_path, exist_ok=True)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training complete!")


def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description="Train Visual Backstory Generation Model")
    
    # Data arguments
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing processed data and features")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of training steps for LR warmup")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Steps between logging")
    parser.add_argument("--generation_epochs", type=int, default=1,
                        help="Epochs between generation samples")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="Weight for KL divergence loss")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()
